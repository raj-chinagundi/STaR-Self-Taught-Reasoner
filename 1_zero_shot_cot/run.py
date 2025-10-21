"""
Zero-Shot Chain-of-Thought inference on GSM8k.

Evaluates a language model using zero-shot CoT prompting.
No training required - purely prompt-based reasoning.
GPU-optimized with batch processing.

Usage:
    python run.py --model meta-llama/Llama-3.2-3B-Instruct --batch_size 64
    python run.py --save_results  # Save detailed results to JSON
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils.common import load_gsm8k, extract_final_answer, extract_answer_from_response, is_correct


def zero_shot_cot_prompt(question: str) -> str:
    """Zero-shot chain-of-thought prompt."""
    return f"Q: {question}\nA: Let me think step by step.\n\n"


def generate_batch(model, tokenizer, prompts: list, max_new_tokens: int = 512) -> list:
    """Generate responses for a batch of prompts (GPU-optimized)."""
    inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.01,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only generated tokens for each sample
    batch_responses = []
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    for i, input_len in enumerate(input_lengths):
        generated = outputs[i][input_len:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        batch_responses.append(response)
    
    return batch_responses


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot CoT on GSM8k")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference (default 64 for A100 80GB)")
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    print(f"Loading GSM8k {args.split} split...")
    data = load_gsm8k(args.split)
    if args.max_samples:
        data = data[:args.max_samples]
    
    correct = 0
    total = len(data)
    results = []
    
    # Process in batches for GPU efficiency
    for batch_start in range(0, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total)
        batch_data = data[batch_start:batch_end]
        
        # Prepare batch
        batch_questions = [ex["question"] for ex in batch_data]
        batch_prompts = [zero_shot_cot_prompt(q) for q in batch_questions]
        batch_ground_truths = [extract_final_answer(ex["answer"]) for ex in batch_data]
        
        # Generate batch responses
        batch_responses = generate_batch(model, tokenizer, batch_prompts)
        
        # Process results
        for j, (question, ground_truth, response) in enumerate(zip(batch_questions, batch_ground_truths, batch_responses)):
            i = batch_start + j
            prediction = extract_answer_from_response(response)
            is_correct_answer = is_correct(prediction, ground_truth)
            
            if is_correct_answer:
                correct += 1
            
            # Store result if saving
            if args.save_results:
                results.append({
                    "id": i + 1,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "correct": is_correct_answer,
                    "response": response
                })
        
        # Progress update
        print(f"Progress: {batch_end}/{total} | Accuracy: {correct/batch_end:.3f} | Batch size: {len(batch_data)}")
    
    accuracy = correct / total
    print(f"\n{'='*60}")
    print(f"Final Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"{'='*60}")
    
    # Save results if requested
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.replace("/", "_").replace("-", "_")
        filename = f"results/zero_shot_cot_{model_name}_{args.split}_{timestamp}.json"
        
        os.makedirs("results", exist_ok=True)
        
        results_data = {
            "metadata": {
                "model_name": args.model,
                "split": args.split,
                "total_samples": total,
                "correct": correct,
                "accuracy": accuracy,
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }
        
        with open(filename, "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")


if __name__ == "__main__":
    main()

