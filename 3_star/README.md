# STaR: Self-Taught Reasoner

Implementation of the Self-Taught Reasoner (STaR) method - iterative self-improvement through reasoning bootstrapping where the model learns from its own successful problem-solving attempts.

## Method Overview

STaR improves model reasoning through iterative bootstrapping:

1. **Generate:** Model creates rationales for training problems using few-shot prompting
2. **Filter:** Keep rationales that lead to correct answers
3. **Rationalize:** For wrong answers, provide correct answer as hint and regenerate
4. **Train:** Fine-tune base model on collected successful rationales
5. **Repeat:** Use improved model for next iteration

Each iteration uses a progressively better model for generation, creating a virtuous cycle of self-improvement.

## Code Structure

```
3_star/
├── create_data.py             # Generates self-reasoning training data with rationalization
├── train.py                   # Fine-tunes model on self-generated rationales
├── evaluate.py                # Evaluates model after each iteration
├── data/
│   ├── star_iter1_*.jsonl     # Iteration 1 training data
│   ├── star_iter2_*.jsonl     # Iteration 2 training data
│   └── ...
└── results/
    ├── model_star_iter1/      # Iteration 1 checkpoint
    ├── model_star_iter2/      # Iteration 2 checkpoint
    └── *.json                 # Evaluation results per iteration
```

### `create_data.py`
Two-stage generation process:
- **Stage 1:** Generate rationales with K few-shot examples; keep if answer correct
- **Stage 2:** For failures, provide correct answer as hint; regenerate and keep if now correct
- Uses batch processing for GPU efficiency
- Outputs JSONL with successful rationales for training

### `train.py`
Fine-tunes model on self-generated data:
- Loads base model (always restart from original, not previous iteration)
- Trains on filtered successful rationales
- Saves iteration checkpoint

### `evaluate.py`
Evaluates iteration model:
- Loads trained iteration model
- Runs inference on test set with batch processing
- Computes accuracy metrics
- Saves detailed results

## Usage

### Manual Step-by-Step (Recommended for Control)

```bash
# ============== ITERATION 1 ==============
# Generate training data from base model
python create_data.py \
  --iteration 1 \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --k 10 \
  --batch_size 32 \
  --output data/star_iter1_train.jsonl

# Train iteration 1 model
python train.py \
  --iteration 1 \
  --train_file data/star_iter1_train.jsonl \
  --output_dir results/model_star_iter1 \
  --epochs 3

# Evaluate iteration 1
python evaluate.py \
  --iteration 1 \
  --model results/model_star_iter1 \
  --save_results

# ============== ITERATION 2 ==============
# Generate data using iteration 1 model (improved reasoning)
python create_data.py \
  --iteration 2 \
  --model_name results/model_star_iter1 \
  --k 10 \
  --batch_size 32 \
  --output data/star_iter2_train.jsonl

# Train iteration 2 from BASE model (not iter1)
python train.py \
  --iteration 2 \
  --train_file data/star_iter2_train.jsonl \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --output_dir results/model_star_iter2 \
  --epochs 3

# Evaluate iteration 2
python evaluate.py \
  --iteration 2 \
  --model results/model_star_iter2 \
  --save_results

# Repeat for iteration 3, 4, etc...
```

### Quick Test on Subset

```bash
# Test on 100 examples only
python create_data.py --iteration 1 --max_samples 100
python train.py --iteration 1
python evaluate.py --iteration 1 --max_samples 100
```

## Command-Line Arguments

### `create_data.py`
- `--iteration`: Iteration number (default: `1`)
- `--model_name`: Model for generation (default: base model for iter1, previous iter for iter2+)
- `--k`: Number of few-shot examples (default: `10`)
- `--batch_size`: Batch size for generation (default: `32`)
- `--output`: Output JSONL file (default: `data/star_iter{N}_train.jsonl`)
- `--max_samples`: Limit training samples (default: `None` = all 7,473)

### `train.py`
- `--iteration`: Iteration number (default: `1`)
- `--train_file`: Training data path (default: `data/star_iter{N}_train.jsonl`)
- `--base_model`: Base model to train from (default: `meta-llama/Llama-3.2-3B-Instruct`)
- `--output_dir`: Save location (default: `results/model_star_iter{N}`)
- `--epochs`: Training epochs (default: `3`)
- `--batch_size`: Per-device batch size (default: `8`)
- `--lr`: Learning rate (default: `2e-5`)

### `evaluate.py`
- `--iteration`: Iteration number (default: `1`)
- `--model`: Model path (default: `results/model_star_iter{N}`)
- `--split`: Dataset split (default: `test`)
- `--batch_size`: Inference batch size (default: `64`)
- `--max_samples`: Limit evaluation (default: `None`)
- `--save_results`: Save detailed JSON (default: `False`)

## STaR Algorithm Details

**Key Insight:** Always train from the original base model, not from previous iteration:
- **Generation Model:** M_{i-1} (previous iteration's trained model)
- **Training Base:** M_0 (original pretrained model)
- **Why:** Prevents error accumulation and overfitting from compounding iterations

**Rationalization:** When model gets wrong answer:
1. Provide correct answer as hint in prompt
2. Ask model to generate reasoning leading to that answer
3. If now correct, include in training data
4. This helps model learn reasoning paths it couldn't discover alone

## Hardware Requirements

**Per Iteration:**

**Minimum:**
- GPU: 24GB VRAM (RTX 3090, RTX 4090)
- RAM: 32GB
- Time: ~12-15 hours per iteration

**Recommended:**
- GPU: A100 40GB or 80GB
- RAM: 64GB
- Time: ~8-10 hours per iteration

**Full 3-Iteration STaR:**
- Total time: ~24-30 hours (A100)
- Total time: ~36-45 hours (24GB GPU)

**Breakdown per iteration:**
- Data generation: ~3-4 hours
- Training: ~2-3 hours
- Evaluation: ~30 minutes

## Expected Results

Performance on GSM8k test set (1,319 examples):

| Iteration | Accuracy | Improvement | Notes |
|-----------|----------|-------------|-------|
| Base (Zero-Shot) | 10-15% | - | Starting baseline |
| Vanilla SFT | 30-40% | +20-25% | Human rationales |
| STaR Iter 1 | 35-40% | +5-10% | First self-improvement |
| STaR Iter 2 | 40-45% | +5% | Second bootstrap |
| STaR Iter 3 | 45-50% | +5% | Diminishing returns |

**Why STaR Works:**
- Model generates diverse reasoning strategies
- Learns from its own successful problem-solving
- Discovers reasoning paths not in human-written data
- Iterative improvement through bootstrapping

## Data Generation Statistics

Typical ratios per iteration:
- **Stage 1 Success:** ~30-40% generate correct answer directly
- **Stage 2 Success:** ~20-30% succeed with hint rationalization
- **Total Training Data:** ~50-70% of original 7,473 examples (~3,500-5,200)

Later iterations have higher Stage 1 success as model improves.

- `data/star_iter{N}_{timestamp}.jsonl` - Training datasets
- `results/model_iter{N}_{timestamp}/` - Trained models

#STEP 1
python create_data.py --batch_size 256 --k 3 

#STEP2
python train_2.py --train_file data/star_iter1_20251020_033715.jsonl --output_dir results/model_star_iter1 --batch_size 32 

#STEP3
python evaluate_2.py --model results/model_star_iter2 --save_results

