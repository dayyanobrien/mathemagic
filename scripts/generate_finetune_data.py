"""Generate the fine-tuning data used in Section 4.8 of the paper.

For every transformation this writes three kinds of 8-shot datasets, each with
a train split (2,500 items) and a test split (625 items) whose expressions do
not overlap:

* ``format_dataset_excl_<T>``  - leave-one-out: every transformation except T
* ``format_dataset_incl_<T>``  - only transformation T
* ``format_dataset_combined``  - all nine, evenly distributed

Each line holds chat ``messages`` (system / user / assistant) ready for
supervised fine-tuning, plus ``messages_original`` with the same prompt under
ordinary arithmetic.

The files shipped in ``data/finetune/`` are the exact ones used in the paper.
They were produced by an unseeded run, so this script (now seeded by
``--seed``) produces different but equivalent splits.

This script keeps its own copy of the transformation code because it differs
in small ways from ``mathemagic/transforms.py`` (e.g. trigonometric
substitution only replaces the first number, and base conversion drops
trailing zero digits). Use ``mathemagic/`` for evaluation.

Usage:
    python scripts/generate_finetune_data.py --out-dir data/finetune_new --seed 0
"""

import argparse
import json
import math
import os
import random
import re

def load_expressions():
    """Return (train, test) GSM8K calculator expressions, shuffled as in the paper."""
    from datasets import load_dataset

    print("Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main")
    pattern = re.compile(r"<<([^<>]+)>>")
    train = [m for item in dataset["train"] for m in pattern.findall(item["answer"])]
    test = [m for item in dataset["test"] for m in pattern.findall(item["answer"])]
    shuffler = random.Random('hplt')
    shuffler.shuffle(train)
    shuffler.shuffle(test)
    print(f"Found {len(train)} training expressions and {len(test)} test expressions.")
    return train, test


# --- Transformation Definitions ---

# Group transformations by category.
transformation_groups = {
    'fundamental': ['swap_symbols'],
    'numeric': ['swap_digits', 'reverse_place_values', 'rotate_digits'],
    'operations': ['add_constant', 'negate_result', 'modular_arithmetic'],
    'advanced': ['trig_substitution', 'change_base']
}
# Create a flat list of all available transformations.
all_transformations = [t for group in transformation_groups.values() for t in group]


# --- Core Transformation Logic ---

def transform_expression(expression, seed=42, transformations=None):
    """
    Transform a mathematical expression using one or more specified transformations.

    Args:
        expression (str): Mathematical expression with an '=' sign.
        seed (int): Random seed for consistency.
        transformations (str, list, optional): A single transformation (str) or a list of
                                              transformations to apply. If None, one is
                                              randomly selected from all_transformations.

    Returns:
        tuple: (original_lhs, new_result, transformation_log)
    """
    rng = random.Random(seed)
    sides = expression.split('=')
    if len(sides) != 2:
        return expression, None, ["Invalid expression format"]

    original_lhs, original_rhs = sides
    lhs = original_lhs.strip()

    # Determine which transformations to apply
    if transformations is None:
        # If no transformation is specified, pick one at random
        transforms_to_apply = [rng.choice(all_transformations)]
    elif isinstance(transformations, str):
        # If a single transformation is passed as a string
        transforms_to_apply = [transformations]
    else:
        # If a list of transformations is passed, use it directly
        transforms_to_apply = transformations

    # Precompute transformation parameters for consistency across applications
    transformation_choices = {
        'op_mapping': {op: rng.choice(['+', '-', '*', '/', '%', '//']) for op in ['+', '-', '*', '/']},
        'digit_swap': tuple(rng.sample(range(10), 2)),
        'base': rng.randint(2, 12),
        'constant': rng.choice([i for i in range(-9, 10) if i != 0]),
        'modulo': rng.choice([2, 3, 5, 7, 10]),
        'trig_func': rng.choice(['math.sin', 'math.cos']),
        'trig_angle': rng.choice([0, 30, 45, 60, 90, 180, 270, 360]),
        'rotation': rng.randint(2, 5)
    }

    transformed_expr = lhs
    transformation_log = []
    result_transformations = []

    # Apply each transformation in the sequence
    for transform in transforms_to_apply:
        if transform == 'swap_symbols':
            transformed_expr, swap_log = apply_symbol_swap(transformed_expr, transformation_choices['op_mapping'])
            transformation_log.append(swap_log)
        elif transform == 'swap_digits':
            d1, d2 = transformation_choices['digit_swap']
            digit_map = {str(d1): str(d2), str(d2): str(d1)}
            transformed_expr = apply_digit_swap_with_map(transformed_expr, digit_map)
            transformation_log.append(f"Swapped digits: exchanged {d1} and {d2}")
            # Digits are swapped in the displayed result too.
            result_transformations.append(lambda r, dm=digit_map: apply_digit_map_to_result(r, dm))
        elif transform == 'change_base':
            base = transformation_choices['base']
            transformation_log.append(f"Changed numeral system to base {base}")
            result_transformations.append(lambda r, b=base: float_to_base_n(r, b) if isinstance(r, float) else int_to_base_n(r, b))
        elif transform == 'negate_result':
            transformed_expr = f"-({transformed_expr})"
            transformation_log.append("Negated the expression")
        elif transform == 'add_constant':
            c = transformation_choices['constant']
            transformed_expr = apply_constant_addition(transformed_expr, c)
            transformation_log.append(f"Added constant: increased the expression by {c}")
        elif transform == 'modular_arithmetic':
            m = transformation_choices['modulo']
            transformed_expr = f"({transformed_expr})%{m}"
            transformation_log.append(f"Applied modulo: result modulo {m}")
        elif transform == 'reverse_place_values':
            transformed_expr = apply_place_value_reversal(transformed_expr)
            transformation_log.append("Reversed place values: e.g., 123 becomes 321")
            result_transformations.append(lambda r: apply_place_value_reversal_to_result(r))
        elif transform == 'rotate_digits':
            rot = transformation_choices['rotation']
            transformed_expr = apply_digit_rotation(transformed_expr, rot)
            transformation_log.append(f"Rotated digits: by {rot} position(s)")
            result_transformations.append(lambda r, rot=rot: apply_digit_rotation_to_result(r, -rot))
        elif transform == 'trig_substitution':
            func = transformation_choices['trig_func']
            angle = transformation_choices['trig_angle']
            transformed_expr, details = apply_trig_substitution(transformed_expr, func, angle)
            if details:
                transformation_log.append(f"Trigonometric substitution: {details}")

    try:
        safe_globals = {"__builtins__": {"abs": abs, "int": int, "float": float, "pow": pow, "round": round}, "math": math}
        safe_globals.update(math.__dict__)
        new_result = eval(transformed_expr, safe_globals)
        # Apply any transformations that modify the final result
        for func in result_transformations:
            new_result = func(new_result)
        return original_lhs, new_result, transformation_log
    except ZeroDivisionError:
        return original_lhs, None, transformation_log + ["Error: Division by zero"]
    except Exception as e:
        return original_lhs, None, transformation_log + [f"Evaluation error: {e}"]

def int_to_base_n(n, base):
    """Convert an integer n to a string representation in the given base."""
    if n == 0: return "0"
    if n < 0:
        return '-' + int_to_base_n(-n, base)
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n < base:
        return digits[n]
    res = ""
    while n:
        res = digits[n % base] + res
        n //= base
    return res

def float_to_base_n(n, base, precision=6):
    """Convert a floating-point number to a string representation in the given base."""
    if n < 0:
        return '-' + float_to_base_n(-n, base, precision)
    
    integer_part = int(n)
    integer_result = int_to_base_n(integer_part, base)
    
    fractional_part = n - integer_part
    if fractional_part == 0:
        return integer_result
        
    fractional_result = ""
    for _ in range(precision):
        fractional_part *= base
        digit = int(fractional_part)
        fractional_result += "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[digit]
        fractional_part -= digit
        if fractional_part == 0:
            break
    
    return f"{integer_result}.{fractional_result}"

def apply_symbol_swap(expression, op_mapping):
    """Flexibly swap mathematical operators with different operations."""
    relevant_ops = [op for op in op_mapping if op in expression]
    if not relevant_ops:
        return expression, "Operator mapping: " + str(op_mapping)
    swap_log = "Operator mapping: " + str(op_mapping)
    temp_expr = expression
    # Use placeholders to avoid replacing parts of operators (e.g., '-' in '//')
    for i, op in enumerate(relevant_ops):
        temp_expr = temp_expr.replace(op, f"§OP{i}§")
    for i, op in enumerate(relevant_ops):
        temp_expr = temp_expr.replace(f"§OP{i}§", op_mapping[op])
    return temp_expr, swap_log

def apply_place_value_reversal(expression):
    def reverse_place_values(match):
        num_str = match.group(0)
        if '.' in num_str:
            int_part, dec_part = num_str.split('.')
            return str(int(int_part[::-1])) + '.' + dec_part
        return str(int(num_str[::-1]))
    return re.sub(r'\b\d+(\.\d+)?\b', reverse_place_values, expression)

def apply_digit_rotation(expression, rotation=1):
    def rotate_digits(match):
        num_str = match.group(0)
        if '.' in num_str:
            int_part, dec_part = num_str.split('.')
            if len(int_part) <= 1: return num_str
            rotation_amount = rotation % len(int_part)
            return int_part[-rotation_amount:] + int_part[:-rotation_amount] + '.' + dec_part
        if len(num_str) <= 1: return num_str
        rotation_amount = rotation % len(num_str)
        return num_str[-rotation_amount:] + num_str[:-rotation_amount]
    return re.sub(r'\b\d+(\.\d+)?\b', rotate_digits, expression)

def apply_digit_swap_with_map(expression, digit_mapping):
    """Swaps digits using temporary placeholders to avoid chain-swapping."""
    result = expression
    # First pass: replace digits with placeholders
    for digit in digit_mapping:
        placeholder = f"§DIGIT{digit}§"
        result = result.replace(digit, placeholder)
    # Second pass: replace placeholders with the target digits
    for digit in digit_mapping:
        placeholder = f"§DIGIT{digit}§"
        result = result.replace(placeholder, digit_mapping[digit])
    return result

def apply_constant_addition(expression, constant):
    return f"({expression})+{constant}"

def coerce_number_str(s: str):
    """Converts a string to an int or float, dropping trailing .0"""
    try:
        if '.' in s:
            if float(s).is_integer():
                return int(float(s))
            return float(s)
        return int(s)
    except (ValueError, TypeError):
        return s # Return original string if conversion fails

def apply_digit_map_to_result(result, digit_mapping):
    s = str(coerce_number_str(str(result)))
    # Use placeholders to avoid issues where a replacement is also a key
    for d in digit_mapping:
        placeholder = f"§TMP{d}§"
        s = s.replace(d, placeholder)
    for d, rep in digit_mapping.items():
        placeholder = f"§TMP{d}§"
        s = s.replace(placeholder, rep)
    return coerce_number_str(s)

def apply_place_value_reversal_to_result(result):
    s = str(coerce_number_str(str(result)))
    negative = s.startswith('-')
    if negative: s = s[1:]
    
    if '.' in s:
        ip, dp = s.split('.', 1)
        core = ip[::-1] + '.' + dp
    else:
        core = s[::-1]
        
    if negative: core = '-' + core
    return coerce_number_str(core)

def apply_digit_rotation_to_result(result, rotation=1):
    s = str(coerce_number_str(str(result)))
    negative = s.startswith('-')
    if negative: s = s[1:]

    if '.' in s:
        ip, dp = s.split('.', 1)
        if len(ip) > 1:
            r = rotation % len(ip)
            rotated_int = ip[-r:] + ip[:-r]
            core = rotated_int + '.' + dp
        else:
            core = s
    else:
        if len(s) > 1:
            r = rotation % len(s)
            core = s[-r:] + s[:-r]
        else:
            core = s
            
    if negative: core = '-' + core
    return coerce_number_str(core)

def apply_trig_substitution(expression, trig_func, angle):
    angle_rad = math.radians(angle)
    # Find the first number to substitute
    match = re.search(r'\b\d+\.?\d*\b', expression)
    if match:
        original_num = match.group(0)
        # Replace only the first occurrence
        result = expression.replace(original_num, f"{trig_func}({angle_rad})", 1)
        detail = f"replaced first number ({original_num}) with {trig_func}({angle}°)"
        return result, detail
    return expression, None

def original_answer(expression):
    """The answer under ordinary arithmetic (right-hand side), or None."""
    safe_globals = {"__builtins__": {"abs": abs, "int": int, "float": float, "pow": pow, "round": round}, "math": math}
    safe_globals.update(math.__dict__)
    sides = expression.split('=')
    try:
        return eval((sides[1] if len(sides) == 2 else sides[0]).strip(), safe_globals)
    except Exception:
        return None


# --- Main Dataset Generation Function ---

def generate_and_save_dataset(
    test_expressions,
    train_expressions,
    transformations,    
    size,
    n_shot,
    filename,
    metadata_info=None,
    outer_seed=None,
    evenly_distribute_transforms=False,
    include_both_versions=False
    ):
    """
    Generates a dataset of n-shot problems and saves it to a JSONL file.

    Args:
        test_expressions (list): Expressions for the main problem.
        train_expressions (list): Expressions for n-shot examples.
        transformations (list): Possible transformation types to choose from.
        size (int): Total number of problems to generate.
        n_shot (int): Number of examples per problem.
        filename (str): Output file name.
        metadata_info (dict, optional): Extra metadata to save.
        outer_seed (int, optional): Seed for reproducibility.
        evenly_distribute_transforms (bool): If True, evenly distributes problems
                                           across all available transformations.
        include_both_versions (bool): If True, includes both transformed and original answers.
    """
    print(f"Generating a dataset of {size} problems for '{filename}'...")
    metadata_info = dict(metadata_info or {})
    rng = random.Random(outer_seed)

    system_rule = "You are a helpful AI assistant in a world where mathematical notation has changed. Use only the provided context. Always answer using decimal numbers only."
    formatting_instruction = "Instruction:\nSolve the problem and present only the final result as LaTeX \\boxed{answer}.\n"

    generated_count = 0
    with open(filename, 'w') as f:
        while generated_count < size:
            # For each problem, select a transformation rule
            if evenly_distribute_transforms:
                # Evenly distribute across all transformations
                transformation_type = transformations[generated_count % len(transformations)]
                metadata_info["transformation_type"] = transformation_type
            else:
                # Pick one transformation from the provided list for this problem
                transformation_type = rng.choice(transformations)
                metadata_info["transformation_type"] = transformation_type

            # Keep trying until we get a successful transformation
            problem_lhs = None
            transformed_answer = None
            target_expr = None
            item_seed = None
            
            while transformed_answer is None:
                target_expr = rng.choice(test_expressions)
                item_seed = rng.randint(0, 2**32 - 1)

                # Generate the Target Problem - always use same transformation to get same question
                problem_lhs, transformed_answer, _ = transform_expression(
                    target_expr, seed=item_seed, transformations=transformation_type
                )
            
            # Always calculate the original answer for comparison
            original_target_answer = original_answer(target_expr)
            if original_target_answer is None:
                continue

            # Keep trying until we get enough valid examples
            examples_transformed = []
            examples_original = []
            chosen_example_exprs = []
            
            while len(examples_transformed) < n_shot:
                # Randomly sample n-shot examples (need to sample more to account for failures)
                example_pool = [expr for expr in train_expressions if expr != target_expr]
                if len(example_pool) < n_shot:
                    print(f"Warning: Not enough unique training examples ({len(example_pool)}) to sample {n_shot}. This shouldn't happen often.")
                    # Try with a different target expression
                    target_expr = rng.choice(test_expressions)
                    item_seed = rng.randint(0, 2**32 - 1)
                    problem_lhs, transformed_answer, _ = transform_expression(
                        target_expr, seed=item_seed, transformations=transformation_type
                    )
                    if transformed_answer is None:
                        continue
                    original_target_answer = original_answer(target_expr)
                    if original_target_answer is None:
                        continue
                    # Reset examples if we changed the target
                    examples_transformed = []
                    examples_original = []
                    chosen_example_exprs = []
                    continue
                
                # Sample more examples than needed to account for failures
                remaining_needed = n_shot - len(examples_transformed)
                sample_size = min(len(example_pool), remaining_needed * 2)  # Sample up to 2x what we need
                
                # Ensure we don't re-sample already chosen examples
                available_pool = [expr for expr in example_pool if expr not in chosen_example_exprs]
                if len(available_pool) < remaining_needed:
                    available_pool = example_pool  # Reset if we've exhausted unique examples
                    
                chosen_new_examples = rng.sample(available_pool, min(sample_size, len(available_pool)))

                for example_expr_orig in chosen_new_examples:
                    if len(examples_transformed) >= n_shot:
                        break
                        
                    # Generate the transformed question to keep examples consistent
                    ex_lhs, transformed_ans, _ = transform_expression(
                        example_expr_orig, seed=item_seed, transformations=transformation_type
                    )
                    if transformed_ans is None:
                        continue
                    
                    # Get the original answer
                    original_ex_ans = original_answer(example_expr_orig)
                    if original_ex_ans is None:
                        continue
                        
                    # Both answers are valid, add to lists
                    transformed_ans = coerce_number_str(str(transformed_ans))
                    original_ex_ans = coerce_number_str(str(original_ex_ans))
                    
                    examples_transformed.append(f"{ex_lhs} = \\boxed{{{transformed_ans}}}")
                    examples_original.append(f"{ex_lhs} = \\boxed{{{original_ex_ans}}}")
                    chosen_example_exprs.append(example_expr_orig)

            # Assemble the Final JSON object with transformed version
            messages_transformed = [{'role': 'system', 'content': f'{system_rule.strip()}\n{formatting_instruction.strip()}'}]
            examples_string_transformed = "Examples:\n" + "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples_transformed))
            problem_string = f"\n\nProblem:\n{problem_lhs} = \\boxed{{?}}"
            final_input_transformed = f"{examples_string_transformed}{problem_string}"
            messages_transformed.append({'role': 'user', 'content': final_input_transformed.strip()})
            
            transformed_target_answer = coerce_number_str(str(transformed_answer))
            messages_transformed.append({'role': 'assistant', 'content': f"\\boxed{{{transformed_target_answer}}}"})

            data_entry = {"messages": messages_transformed, **metadata_info}
            
            # If including both versions, also create the original version
            if include_both_versions:
                messages_original = [{'role': 'system', 'content': f'{system_rule.strip()}\n{formatting_instruction.strip()}'}]
                examples_string_original = "Examples:\n" + "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples_original))
                final_input_original = f"{examples_string_original}{problem_string}"
                messages_original.append({'role': 'user', 'content': final_input_original.strip()})
                
                original_target_answer_str = coerce_number_str(str(original_target_answer))
                messages_original.append({'role': 'assistant', 'content': f"\\boxed{{{original_target_answer_str}}}"})
                
                data_entry["messages_original"] = messages_original
            
            f.write(json.dumps(data_entry) + '\n')
            generated_count += 1

            if generated_count % 250 == 0:
                print(f"  ... Generated {generated_count}/{size} problems for {filename}")

    print(f"\nDataset generation complete. Saved to '{filename}'.")


def generate_train_test_split_datasets(
    test_expressions,
    train_expressions,
    transformations,
    size,
    n_shot,
    base_filename,
    metadata_info=None,
    outer_seed=None,
    evenly_distribute_transforms=False,
    test_split_ratio=0.2
    ):
    """
    Generate separate train and test datasets with no expression crossover.
    Each file will contain both transformed and original versions in the same JSON object.
    
    Args:
        test_expressions (list): All available test expressions.
        train_expressions (list): All available train expressions.
        transformations (list): Transformations to apply.
        size (int): Total number of problems to generate (will be split).
        n_shot (int): Number of examples per problem.
        base_filename (str): Base filename (will add _train.jsonl and _test.jsonl).
        metadata_info (dict): Extra metadata.
        outer_seed (int): Random seed.
        evenly_distribute_transforms (bool): Whether to evenly distribute transformations.
        test_split_ratio (float): Fraction of data for test set.
    """
    rng = random.Random(outer_seed)
    
    # Calculate split sizes
    test_size = int(size * test_split_ratio)
    train_size = size - test_size
    
    # Split expressions ensuring no crossover
    all_test_exprs = test_expressions.copy()
    all_train_exprs = train_expressions.copy()
    
    # Shuffle both lists
    rng.shuffle(all_test_exprs)
    rng.shuffle(all_train_exprs)
    
    # Split test expressions
    test_split_exprs = all_test_exprs[:test_size] if len(all_test_exprs) >= test_size else all_test_exprs
    remaining_test_exprs = all_test_exprs[test_size:]
    
    # Split train expressions to ensure no overlap in n-shot examples
    train_exprs_for_test = all_train_exprs[:len(all_train_exprs)//2]
    train_exprs_for_train = all_train_exprs[len(all_train_exprs)//2:]
    
    # For train set, use remaining test expressions + some train expressions
    # but ensure no overlap with test split
    train_split_exprs = remaining_test_exprs + train_exprs_for_train
    
    # Ensure we have enough expressions for both splits
    if len(test_split_exprs) < test_size:
        print(f"Warning: Not enough test expressions. Requested {test_size}, got {len(test_split_exprs)}")
        test_size = len(test_split_exprs)
    
    if len(train_split_exprs) < train_size:
        print(f"Warning: Not enough train expressions. Requested {train_size}, got {len(train_split_exprs)}")
        train_size = len(train_split_exprs)
    
    # Generate train dataset with both versions
    train_filename = base_filename.replace('.jsonl', '_train.jsonl')
    train_metadata = dict(metadata_info or {})
    train_metadata['split'] = 'train'
    
    generate_and_save_dataset(
        test_expressions=train_split_exprs[:train_size],
        train_expressions=train_exprs_for_train,
        transformations=transformations,
        size=train_size,
        n_shot=n_shot,
        filename=train_filename,
        metadata_info=train_metadata,
        outer_seed=rng.randint(0, 2**32 - 1),
        evenly_distribute_transforms=evenly_distribute_transforms,
        include_both_versions=True
    )
    
    # Generate test dataset with both versions
    test_filename = base_filename.replace('.jsonl', '_test.jsonl')
    test_metadata = dict(metadata_info or {})
    test_metadata['split'] = 'test'
    
    generate_and_save_dataset(
        test_expressions=test_split_exprs,
        train_expressions=train_exprs_for_test,
        transformations=transformations,
        size=test_size,
        n_shot=n_shot,
        filename=test_filename,
        metadata_info=test_metadata,
        outer_seed=rng.randint(0, 2**32 - 1),
        evenly_distribute_transforms=evenly_distribute_transforms,
        include_both_versions=True
    )


# --- Script Execution ---

def main():
    parser = argparse.ArgumentParser(description="Generate MatheMagic fine-tuning datasets.")
    parser.add_argument("--out-dir", default="data/finetune_new")
    parser.add_argument("--seed", type=int, default=0, help="Seed for splits and sampling.")
    parser.add_argument("--size", type=int, default=3125, help="Items per dataset before the 80/20 split.")
    parser.add_argument("--n-shot", type=int, default=8)
    args = parser.parse_args()

    math_expr_train, math_expr_test = load_expressions()
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = random.Random(args.seed)

    def make(transformations, name, metadata, evenly=False):
        generate_train_test_split_datasets(
            test_expressions=math_expr_test,
            train_expressions=math_expr_train,
            transformations=transformations,
            size=args.size,
            n_shot=args.n_shot,
            base_filename=os.path.join(args.out_dir, f"{name}.jsonl"),
            metadata_info=metadata,
            outer_seed=seeds.randint(0, 2**32 - 1),
            evenly_distribute_transforms=evenly,
        )

    print("\n--- Leave-one-out datasets (format_dataset_excl_*) ---")
    for t in all_transformations:
        make([x for x in all_transformations if x != t], f"format_dataset_excl_{t}",
             {"generation_mode": "leave-one-out", "excluded_transformation": t})

    print("\n--- Single-transformation datasets (format_dataset_incl_*) ---")
    for t in all_transformations:
        make([t], f"format_dataset_incl_{t}",
             {"generation_mode": "include-only-one", "included_transformation": t})

    print("\n--- Combined dataset (format_dataset_combined) ---")
    make(all_transformations, "format_dataset_combined", {"generation_mode": "combined"}, evenly=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
