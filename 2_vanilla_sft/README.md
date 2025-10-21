# Vanilla Supervised Fine-Tuning (SFT)

Standard supervised fine-tuning approach that trains a language model on GSM8k's original human-written reasoning traces.

## Method Overview

Trains the model to imitate human problem-solving by learning from the dataset's original step-by-step solutions. This establishes a supervised learning baseline before attempting self-improvement via STaR.

## Code Structure

```
2_vanilla_sft/
├── create_data.py             # Converts GSM8k to training format (JSONL)
├── train.py                   # Fine-tunes model on human rationales
├── evaluate.py                # Evaluates trained model performance
├── data/
│   └── train.jsonl            # Training data (7,473 examples)
└── results/
    ├── model/                 # Trained model checkpoints
    └── *.json                 # Evaluation results
```

### `create_data.py`
Prepares training data by:
- Loading GSM8k train split (7,473 examples)
- Extracting question and answer (with reasoning) pairs
- Saving to JSONL format for training

### `train.py`
Fine-tunes the model:
- Loads base model and tokenizer
- Formats data as conversational turns
- Trains using HuggingFace Trainer with gradient accumulation
- Saves checkpoints to `results/model/`

### `evaluate.py`
Evaluates trained model:
- Loads fine-tuned model
- Runs inference on test set with batch processing
- Extracts numerical answers and computes accuracy
- Optionally saves detailed per-example results

## Usage

### Step 1: Create Training Data

```bash
python create_data.py
```

**Output:** `data/train.jsonl` (7,473 training examples)

Each line contains:
```json
{
  "question": "Natalia sold clips to 48 of her friends...",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n..."
}
```

### Step 2: Train Model

```bash
# Default training (3 epochs, batch_size=8, lr=2e-5)
python train.py

# Custom hyperparameters
python train.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --epochs 5 \
  --batch_size 16 \
  --lr 1e-5 \
  --output_dir results/custom_model
```

**Command-Line Arguments:**
- `--model_name`: Base model to fine-tune (default: `meta-llama/Llama-3.2-3B-Instruct`)
- `--train_file`: Training data path (default: `data/train.jsonl`)
- `--output_dir`: Where to save model (default: `results/model`)
- `--epochs`: Training epochs (default: `3`)
- `--batch_size`: Per-device batch size (default: `8`)
- `--lr`: Learning rate (default: `2e-5`)
- `--grad_accum`: Gradient accumulation steps (default: `4`)

**Output:** Trained model saved to `results/model/`

### Step 3: Evaluate Model

```bash
# Basic evaluation on test set
python evaluate.py

# Save detailed results
python evaluate.py --save_results

# Evaluate specific model checkpoint
python evaluate.py --model results/model

# Adjust batch size for your GPU
python evaluate.py --batch_size 32

# Quick test on subset
python evaluate.py --max_samples 100
```

**Command-Line Arguments:**
- `--model`: Path to trained model (default: `results/model`)
- `--split`: Dataset split to evaluate (default: `test`)
- `--batch_size`: Batch size for inference (default: `64`)
- `--max_samples`: Limit evaluation samples (default: `None`)
- `--save_results`: Save detailed JSON results (default: `False`)

**Saved Results:** `results/vanilla_sft_results_model_{split}_{timestamp}.json`


