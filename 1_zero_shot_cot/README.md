# Zero-Shot Chain-of-Thought

Evaluates language model's baseline mathematical reasoning capability using zero-shot chain-of-thought prompting on GSM8k dataset.

## Method Overview

Prompts the model with: `"Q: {question}\nA: Let me think step by step."`

This method requires **no training** - it's purely prompt-based reasoning to establish a baseline before fine-tuning.

## Code Structure

```
1_zero_shot_cot/
├── run.py                     # Main evaluation script with GPU batch processing
└── results/                   # Output directory for evaluation results
    └── *.json                 # Detailed per-example results
```

### `run.py`
Main evaluation script that:
- Loads GSM8k test/train split
- Generates responses using zero-shot CoT prompting
- Extracts and validates final numerical answers
- Computes accuracy metrics
- Saves detailed results to JSON

## Usage

### Basic Evaluation
```bash
# Run on test set (1,319 examples)
python run.py

# Save detailed results to JSON file
python run.py --save_results
```

### Advanced Options
```bash
# Use specific model
python run.py --model meta-llama/Llama-3.2-3B-Instruct

# Adjust batch size for your GPU memory
python run.py --batch_size 32    # For 24GB GPU
python run.py --batch_size 64    # For 40GB+ GPU (default)

# Quick test on subset
python run.py --max_samples 100

# Evaluate on train split instead
python run.py --split train
```

### Command-Line Arguments
- `--model`: HuggingFace model name (default: `meta-llama/Llama-3.2-3B-Instruct`)
- `--split`: Dataset split - `test` or `train` (default: `test`)
- `--batch_size`: Batch size for inference (default: `64`)
- `--max_samples`: Limit number of examples (default: `None` = all)
- `--save_results`: Save detailed JSON results (default: `False`)

## Hardware Requirements

**Minimum:**
- GPU: 24GB VRAM (e.g., RTX 3090, RTX 4090)
- Batch size: 32
- Time: ~45 minutes for full test set

**Recommended:**
- GPU: 40GB+ VRAM (e.g., A100)
- Batch size: 64
- Time: ~30 minutes for full test set

## Output

### Console Output
```
Loading model: meta-llama/Llama-3.2-3B-Instruct
Evaluating on test split...
Batch 1/21 | Running Accuracy: 12.5%
Batch 2/21 | Running Accuracy: 14.1%
...
Final Accuracy: 13.2% (174/1319)
```

### Saved Results (with `--save_results`)
**Location:** `results/zero_shot_cot_{model}_{split}_{timestamp}.json`

**Format:**
```json
{
  "metadata": {
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "split": "test",
    "total_examples": 1319,
    "correct": 174,
    "accuracy": 0.132,
    "timestamp": "20251020_123456"
  },
  "results": [
    {
      "question": "Natalia sold clips...",
      "gold_answer": "48",
      "generated_response": "Let me think step by step...",
      "extracted_answer": "48",
      "correct": true
    },
    ...
  ]
}
```
