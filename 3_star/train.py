"""
Train model on GSM8k original rationales (vanilla SFT).

Usage:
    python train.py --train_file data/train.jsonl --output_dir results/model
"""

import argparse
import jsonlines
import os
from typing import List, Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 1024):
        self.samples: List[Dict[str, str]] = []
        with jsonlines.open(path, 'r') as reader:
            for item in reader:
                if "question" in item and "answer" in item:
                    self.samples.append({
                        "question": item["question"],
                        "answer": item["answer"],
                    })
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        
        # Format: Q: question\nA: answer
        question_text = f"Q: {s['question']}\nA: "
        answer_text = f"{s['answer']}{self.tokenizer.eos_token}"
        full_text = question_text + answer_text
        
        # Tokenize full text
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        
        # Create labels with loss masking for question
        labels = input_ids.clone()
        
        # Find where answer starts
        question_enc = self.tokenizer(
            question_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        question_length = len(question_enc["input_ids"][0])
        
        # Mask question tokens (no loss computed on prompt)
        if question_length < len(labels):
            labels[:question_length] = -100
        else:
            labels[:-1] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser(description="Vanilla SFT training")
    parser.add_argument("--train_file", default="data/train.jsonl")
    parser.add_argument("--output_dir", default="results/model")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    print(f"Loading training data: {args.train_file}")
    train_ds = SFTDataset(args.train_file, tokenizer, max_length=args.max_length)
    print(f"Loaded {len(train_ds)} training examples")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        warmup_steps=100,
        weight_decay=0.01,
        bf16=True,
        remove_unused_columns=False,
        report_to=None,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )
    
    print("Starting training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print(f"✓ Training complete! Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

