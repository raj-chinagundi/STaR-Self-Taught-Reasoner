"""Common utilities for GSM8k experiments."""

import re
from typing import List, Dict
from datasets import load_dataset


def load_gsm8k(split: str) -> List[Dict[str, str]]:
    """Load GSM8k dataset."""
    ds = load_dataset("gsm8k", "main", split=split)
    return [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]


def extract_final_answer(text: str) -> str:
    """Extract final numeric answer from GSM8k format (after ####)."""
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


def extract_answer_from_response(response: str) -> str:
    """Extract numeric answer from model response.
    
    Handles multiple formats:
    - LaTeX boxed: \\boxed{42}
    - GSM8k format: #### 42
    - Fractions: 3/4 -> 0.75
    - Scientific notation: 1.2e+3
    - Fallback: last number in text
    """
    # Handle boxed answers
    boxed_match = re.search(r'\$?\\boxed\{([^}]+)\}\$?', response)
    if boxed_match:
        return boxed_match.group(1).replace(",", "").strip()
    
    # Clean formatting
    text = response.replace("$", "").replace(",", "")
    
    # Look after #### first, then anywhere
    search_text = text.split("####")[-1] if "####" in text else text
    
    # Check for fractions
    fraction_match = re.search(r"(\d+)/(\d+)", search_text)
    if fraction_match:
        num, den = map(int, fraction_match.groups())
        if den != 0:
            return str(num / den)
    
    # Find numbers (including negatives, decimals, scientific notation)
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", search_text)
    return nums[-1] if nums else ""


def is_correct(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth (numeric comparison)."""
    def _clean(s: str) -> str:
        return s.replace(",", "").replace("$", "").strip()
    
    pred_nums = re.findall(r"-?\d+(?:\.\d+)?", _clean(predicted))
    gt_nums = re.findall(r"-?\d+(?:\.\d+)?", _clean(ground_truth))
    
    if not pred_nums or not gt_nums:
        return False
    
    try:
        return abs(float(pred_nums[-1]) - float(gt_nums[-1])) < 1e-6
    except ValueError:
        return False

