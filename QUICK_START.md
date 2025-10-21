# Quick Start

## Setup

```bash
pip install -r requirements.txt
```

## Run Methods

### 1. Zero-Shot CoT (No Training Required)

```bash
cd 1_zero_shot_cot && python run.py --batch_size 64 --save_results
```

### 2. Vanilla SFT

```bash
cd 2_vanilla_sft && python create_data.py && python train.py --batch_size 16 --grad_accum 4 && python evaluate.py --batch_size 64 --save_results
```

### 3. STaR

```bash
cd 3_star && python pipeline.py --batch_size 32 --iterations 3
```

---

**Note**: Commands optimized for A100 80GB GPU. Adjust `--batch_size` if needed.
