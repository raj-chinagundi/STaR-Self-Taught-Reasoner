# STaR: Self-Taught Reasoner

Implementation of the Self-Taught Reasoner (STaR) methodology for improving language model reasoning through iterative self-training on GSM8k mathematical reasoning dataset.

# Flow Diagram
 ![flow diagram of STaR](https://pbs.twimg.com/media/G2_p5phbwAEZRY4?format=jpg&name=large)

## Overview

This repository implements three progressive approaches to mathematical reasoning:

1. **Zero-Shot CoT** - Baseline prompt-based reasoning without training
2. **Vanilla SFT** - Standard supervised fine-tuning on human-written solutions
3. **STaR** - Iterative self-improvement through reasoning bootstrapping

## Repository Structure

```
STaR/
├── 1_zero_shot_cot/           # Zero-shot Chain-of-Thought baseline
│   ├── run.py                 # Evaluation script with batch processing
│   └── results/               # Evaluation outputs (JSON)
│
├── 2_vanilla_sft/             # Supervised Fine-Tuning
│   ├── create_data.py         # Converts GSM8k to training format
│   ├── train.py               # Fine-tunes model on human rationales
│   ├── evaluate.py            # Evaluates trained model
│   ├── data/                  # Training data (JSONL)
│   └── results/               # Checkpoints and evaluation results
│
├── 3_star/                    # Self-Taught Reasoner
│   ├── create_data.py         # Generates self-reasoning training data
│   ├── train.py               # Trains on self-generated rationales
│   ├── evaluate.py            # Evaluates iteration models
│   ├── data/                  # STaR iteration datasets
│   └── results/               # Iteration checkpoints and results
│
└── utils/
    └── common.py              # Shared GSM8k utilities and evaluation functions
```

## Methods

### 1. Zero-Shot Chain-of-Thought
Evaluates base model using "Let's think step by step" prompting without any fine-tuning.

### 2. Vanilla Supervised Fine-Tuning
Standard supervised learning on GSM8k's original human-written reasoning traces.

### 3. STaR (Self-Taught Reasoner)
Iterative process where model:
1. Generates rationales for training problems
2. Filters rationales that lead to correct answers
3. Retrains on its own correct reasoning
4. Repeats with improved model


## Detailed Documentation

Each method has comprehensive documentation in its respective folder:
- [1_zero_shot_cot/README.md](1_zero_shot_cot/README.md)
- [2_vanilla_sft/README.md](2_vanilla_sft/README.md)
- [3_star/README.md](3_star/README.md)

## Citation

This implementation is based on:

```bibtex
@article{zelikman2022star,
  title={STaR: Bootstrapping Reasoning With Reasoning},
  author={Zelikman, Eric and Wu, Yuhuai and Mu, Jesse and Goodman, Noah D},
  journal={arXiv preprint arXiv:2203.14465},
  year={2022}
}
```

