"""
STaR dataset generation with few-shot prompting and rationalization.
GPU-optimized with batch processing.

Two-stage generation process:
  Stage 1: Generate rationale with K few-shot examples
           → Keep if answer is correct
  
  Stage 2: For failures, provide correct answer as hint
           → Generate again and keep if now correct
           → Falls back to original rationale if still wrong

Output: JSONL file with successfully generated rationales for training.

Usage:
    python create_data.py --output data/star_train.jsonl --model_name meta-llama/Llama-3.2-3B-Instruct --k 10 --batch_size 32
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from utils.common import load_gsm8k, extract_final_answer, extract_answer_from_response, is_correct

os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/scratch/bchinagu/huggingface")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def fewshot_block(q: str, a: str) -> str:
    return f"Q: {q}\nA: {a}\n\n"


def build_fewshot_prompt(demos: List[Dict[str, str]], target_q: str) -> str:
    prompt = "".join([fewshot_block(d["question"], d["answer"]) for d in demos])
    prompt += f"Q: {target_q}\nA: "
    return prompt


def build_hint_prompt(demos: List[Dict[str, str]], target_q: str, gt_final: str) -> str:
    prompt = "".join([fewshot_block(d["question"], d["answer"]) for d in demos])
    prompt += (
        f"Q: {target_q}\n"
        f"The answer is {gt_final}.\n"
        f"A: Let me work through this step by step to get to the answer {gt_final}.\n\n"
    )
    return prompt


def build_validation_prompt(demos: List[Dict[str, str]], target_q: str, rationale: str) -> str:
    """Build a validation prompt that uses the generated rationale (no hint) and
    explicitly asks the model to produce the final answer after '####'.
    """
    prompt = "".join([fewshot_block(d["question"], d["answer"]) for d in demos])
    prompt += (
        f"Q: {target_q}\n"
        f"A: {rationale}\n"
        f"(Provide only the final number after ####. Do not add extra text.)\n"
        f"Therefore, the answer is #### "
    )
    return prompt


def clean_rationale(text: str) -> str:
    """Remove any echoed hint lines and trailing final-answer segments from a rationale.

    - Drops everything after a '####' if present
    - Removes lines that start with 'The answer is ' (echoed hint)
    - Strips whitespace
    """
    # Remove anything after a final answer marker
    base = text.split("####")[0]
    # Remove echoed hint lines
    lines = [ln for ln in base.splitlines() if not ln.strip().lower().startswith("the answer is ")]
    return ("\n".join(lines)).strip()


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> List[str]:
    """Batched generation for a list of prompts.

    Decodes only generated continuations by slicing each output sequence at its
    true input length (computed from attention_mask sums).
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=True,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Slice per-sample using true input lengths
    attn_lens = attention_mask.sum(dim=1).tolist()
    results: List[str] = []
    for i in range(out.shape[0]):
        gen_part = out[i, attn_lens[i]:]
        text = tokenizer.decode(gen_part, skip_special_tokens=True).strip()
        results.append(text)
    return results


def select_demos(pool: List[Dict[str, str]], k: int, exclude_idx: int, rng: random.Random) -> List[Dict[str, str]]:
    """Select K few-shot demonstrations from the pool, excluding the current index.
    Uses random sampling without replacement for diversity.
    """
    indices = list(range(len(pool)))
    if 0 <= exclude_idx < len(pool):
        indices.pop(exclude_idx)
    rng.shuffle(indices)
    chosen = indices[:k]
    return [pool[i] for i in chosen]


def main():
    ap = argparse.ArgumentParser(description="Few-shot STaR creation (1-try loop) for GSM8k")
    ap.add_argument("--output", required=True, help="Path to output JSONL of successful examples")
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--k", type=int, default=10, help="Number of few-shot demonstrations")
    ap.add_argument("--max_samples", type=int, default=None, help="Limit number of train examples processed")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for inference (default 32 for A100 80GB)")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--deterministic", action="store_true", help="Disable sampling and use temperature=0.0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.deterministic:
        args.temperature = 0.0

    print("Loading GSM8k train split (as pool and target set)...")
    data = load_gsm8k("train")
    if args.max_samples:
        data = data[:args.max_samples]
    total = len(data)
    print(f"Loaded {total} examples")

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    kept_stage1 = 0
    kept_stage2 = 0
    kept_gold = 0
    with open(args.output, "w") as wf:
        B = args.batch_size
        total = len(data)
        for start in tqdm(range(0, total, B), desc="fewshot_star"):
            end = min(start + B, total)
            batch = data[start:end]

            # Precompute gt finals and demos per item
            batch_q = [ex["question"] for ex in batch]
            batch_gt_full = [ex["answer"] for ex in batch]
            batch_gt_final = [extract_final_answer(a) for a in batch_gt_full]
            batch_demos = [select_demos(data, args.k, exclude_idx=start + i, rng=rng) for i in range(end - start)]

            # Stage 1 prompts
            prompts1 = [build_fewshot_prompt(batch_demos[i], batch_q[i]) for i in range(len(batch))]
            resps1 = generate_batch(
                model,
                tokenizer,
                prompts1,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=(not args.deterministic),
            )
            preds1 = [extract_answer_from_response(r) for r in resps1]
            ok1 = [is_correct(preds1[i], batch_gt_final[i]) for i in range(len(batch))]

            # Log Stage 1 results and write successes
            for i in range(len(batch)):
                idx = start + i
                if ok1[i]:
                    kept_stage1 += 1
                    wf.write(json.dumps({
                        "question": batch_q[i],
                        "answer": resps1[i],  # Already in GSM8k format with ####
                        "method": "star_fewshot",
                        "stage": 1,
                    }) + "\n")
                    print(f"[#{idx}] Stage1 ✓ | pred={preds1[i]} | kept")
                else:
                    print(f"[#{idx}] Stage1 ✗ | pred={preds1[i]} | will try Stage2")

            # Prepare Stage 2 only for failures
            fail_indices = [i for i in range(len(batch)) if not ok1[i]]
            if not fail_indices:
                continue

            prompts2 = [build_hint_prompt(batch_demos[i], batch_q[i], batch_gt_final[i]) for i in fail_indices]
            resps2 = generate_batch(
                model,
                tokenizer,
                prompts2,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=(not args.deterministic),
            )
            # Validation prompts using cleaned rationales
            rationales2 = [clean_rationale(resps2[j]) for j in range(len(resps2))]
            prompts2_val = [build_validation_prompt(batch_demos[fail_indices[j]], batch_q[fail_indices[j]], rationales2[j]) for j in range(len(fail_indices))]
            # For validation, use greedy decoding to avoid sampling variance
            resps2_val = generate_batch(
                model,
                tokenizer,
                prompts2_val,
                max_new_tokens=32,
                temperature=0.0,
                do_sample=False,
            )
            preds2_val = [extract_answer_from_response(x) for x in resps2_val]
            ok2 = [is_correct(preds2_val[j], batch_gt_final[fail_indices[j]]) for j in range(len(fail_indices))]

            # Log Stage 2 results and write successes
            for j, i in enumerate(fail_indices):
                idx = start + i
                if ok2[j]:
                    kept_stage2 += 1
                    # Add #### answer to match GSM8k format
                    answer_with_final = rationales2[j] + f"\n#### {preds2_val[j]}"
                    wf.write(json.dumps({
                        "question": batch_q[i],
                        "answer": answer_with_final,
                        "method": "star_fewshot",
                        "stage": 2,
                    }) + "\n")
                    print(f"[#{idx}] Stage2 ✓ | pred={preds2_val[j]} | kept")
                else:
                    # Keep original gold rationale instead of dropping
                    kept_gold += 1
                    wf.write(json.dumps({
                        "question": batch_q[i],
                        "answer": batch_gt_full[i],
                        "method": "star_fewshot",
                        "stage": "gold",
                        "failed": True,
                    }) + "\n")
                    print(f"[#{idx}] Stage2 ✗ | pred={preds2_val[j]} | kept (gold)")

    print(
        f"Few-shot STaR complete. Kept Stage1: {kept_stage1}, "
        f"Stage2 fixes: {kept_stage2}, Kept gold: {kept_gold}, "
        f"Total kept: {kept_stage1 + kept_stage2 + kept_gold} / {total}"
    )


if __name__ == "__main__":
    main()
