"""Building MatheMagic test sets and prompts.

A test set is fully determined by its seed: the seed picks the GSM8K
expressions, each item's transformation parameters and its in-context
examples. The five seeds used in the paper are in ``PAPER_SEEDS``.
"""

import itertools
import json
import random
import re

from .transforms import TRANSFORMATIONS, transform_expression, transforms_to_prompt

PAPER_SEEDS = [42, 68, 100, 789, 1024]
PAPER_SHOTS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
PAPER_ITEMS_PER_TRANSFORMATION = 45  # 9 x 45 = 405 items per seed

# Number of examples generated per item (see ``create_test_set``). This value
# reproduces the paper's test sets.
EXAMPLES_GENERATED_PER_ITEM = 512

SYSTEM_RULE = (
    "You are a helpful AI assistant in a world where mathematical notation has changed. "
    "Use only the provided context. Always answer using decimal numbers only."
)

# Appendix A.3. Variant 1 is the default used in the paper.
INSTRUCTION_VARIANTS = [
    "Instruction:\nPresent your solution in the following format:\n1. Interpret the transformation rules.\n"
    "2. Apply them step by step.\n3. Justify each transformation.\n4. Final Answer: LaTeX \\boxed{answer}.\n",
    "Instruction:\nSolve the problem by:\n1. Explaining the transformation logic.\n"
    "2. Showing the intermediate steps clearly.\n3. Justifying each step briefly.\n4. Final Answer: LaTeX \\boxed{answer}.\n",
    "Instruction:\nSolve the problem and present only the final result as LaTeX \\boxed{answer}.\n",
    "Instruction:\nCompute the correct result. Provide the answer only. Output: \\boxed{answer}\n",
]

SETTINGS = ["inductive", "deductive"]


# --- GSM8K expressions -------------------------------------------------------

def load_gsm8k_expressions():
    """Return ``(train_expressions, test_expressions)`` extracted from GSM8K.

    Expressions are the ``<<...>>`` calculator annotations in GSM8K answers.
    The training list is shuffled with a fixed seed, as in the paper.
    """
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main")
    pattern = re.compile(r"<<([^<>]+)>>")
    train = [m for item in dataset["train"] for m in pattern.findall(item["answer"])]
    test = [m for item in dataset["test"] for m in pattern.findall(item["answer"])]
    random.Random("hplt").shuffle(train)
    return train, test


# --- Test-set generation -----------------------------------------------------

def create_test_set(test_expressions, train_expressions, seed,
                    transformations=TRANSFORMATIONS,
                    items_per_transformation=PAPER_ITEMS_PER_TRANSFORMATION,
                    max_examples=max(PAPER_SHOTS),
                    examples_per_item=EXAMPLES_GENERATED_PER_ITEM):
    """Generate one seeded test set.

    Questions come from the GSM8K *test* split and in-context examples from the
    *train* split. For each transformation, ``items_per_transformation`` items
    are made; each item gets its own seed, shared by its question and its
    examples so they follow the same rule.

    ``max_examples`` only controls how many examples are *stored*.
    ``examples_per_item`` is how many are *generated*; because items draw from
    one shared stream of training expressions, it changes which examples later
    items receive. Leave it at the default to reproduce the paper.
    """
    rng = random.Random(seed)
    shuffled_test = list(test_expressions)
    rng.shuffle(shuffled_test)
    question_cycle = itertools.cycle(shuffled_test)
    shuffled_train = list(train_expressions)
    rng.shuffle(shuffled_train)

    n_examples = min(examples_per_item, len(shuffled_train))
    items = []
    for transformation in transformations:
        example_cycle = itertools.cycle(shuffled_train)
        made = 0
        while made < items_per_transformation:
            original = next(question_cycle)
            item_seed = rng.randint(0, 2**32 - 1)
            problem, answer, log = transform_expression(original, item_seed, transformation)
            if answer is None:
                continue

            examples = []
            attempts = 0
            while len(examples) < n_examples and attempts < len(shuffled_train) * 2:
                ex_problem, ex_answer, _ = transform_expression(next(example_cycle), item_seed, transformation)
                if ex_answer is not None:
                    examples.append({"problem": ex_problem, "answer": ex_answer})
                attempts += 1

            items.append({
                "id": f"{seed}-{transformation}-{made}",
                "seed": seed,
                "transformation_type": transformation,
                "problem": problem,                  # question shown to the model
                "answer": answer,                    # answer under the transformed rules
                "original": original,                # GSM8K expression in ordinary maths
                "hint": transforms_to_prompt(log),   # rule description (deductive setting)
                "examples": examples[:max_examples], # in-context examples, same rule
            })
            made += 1
    return items


# --- I/O ---------------------------------------------------------------------

def save_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Prompts -----------------------------------------------------------------

def build_messages(item, shots, setting="inductive", instruction_variant=1):
    """Build the chat messages for one item.

    Args:
        item: A test item from ``create_test_set`` / the shipped data.
        shots: Number of in-context examples to include.
        setting: ``"inductive"`` (examples only) or ``"deductive"`` (examples
            plus a natural-language description of the rule).
        instruction_variant: 1-4, see ``INSTRUCTION_VARIANTS``.
    """
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}")
    if setting == "deductive" and not item.get("hint"):
        raise ValueError(f"Item {item['id']} has no rule description, so it has no deductive setting.")
    if shots > len(item["examples"]):
        raise ValueError(f"Item {item['id']} only stores {len(item['examples'])} examples; asked for {shots}.")
    instruction = INSTRUCTION_VARIANTS[instruction_variant - 1]
    system = f"{SYSTEM_RULE.strip()}\n{instruction.strip()}"

    user = ""
    examples = item["examples"][:shots]
    if examples:
        user = "Examples:\n" + "\n".join(
            f"{i + 1}. {ex['problem']} = \\boxed{{{ex['answer']}}}" for i, ex in enumerate(examples)
        )
    user += "\nProblem:\n"
    if setting == "deductive":
        user += f"(Hint: {item['hint'].strip()})\n"
    user += f"{item['problem']} = \\boxed{{?}}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user.strip()},
    ]
