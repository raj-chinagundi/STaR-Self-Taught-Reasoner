"""
Create vanilla SFT training data from GSM8k original rationales.

Loads GSM8k train split and saves question-answer pairs to data/train.jsonl.

Usage:
    python create_data.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jsonlines
from tqdm import tqdm
from utils.common import load_gsm8k


def main():
    print("Loading GSM8k train split...")
    train_data = load_gsm8k("train")
    
    output_file = "data/train.jsonl"
    os.makedirs("data", exist_ok=True)
    print(f"Creating vanilla SFT data: {output_file}")
    
    with jsonlines.open(output_file, "w") as writer:
        for example in tqdm(train_data, desc="Processing"):
            writer.write({
                "question": example["question"],
                "answer": example["answer"],
            })
    
    print(f"✓ Created {len(train_data)} training examples")
    print(f"✓ Saved to: {output_file}")


if __name__ == "__main__":
    main()

