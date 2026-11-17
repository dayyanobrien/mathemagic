"""MatheMagic: dynamic, counterfactual maths benchmarks robust to memorisation."""

from .benchmark import (INSTRUCTION_VARIANTS, PAPER_SEEDS, PAPER_SHOTS, SETTINGS, SYSTEM_RULE,
                        build_messages, create_test_set, load_gsm8k_expressions, load_jsonl)
from .scoring import extract_boxed, score_response
from .transforms import TRANSFORMATIONS, transform_expression

__all__ = [
    "INSTRUCTION_VARIANTS", "PAPER_SEEDS", "PAPER_SHOTS", "SETTINGS", "SYSTEM_RULE", "TRANSFORMATIONS",
    "build_messages", "create_test_set", "extract_boxed", "load_gsm8k_expressions", "load_jsonl",
    "score_response", "transform_expression",
]
