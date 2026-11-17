"""Counterfactual arithmetic transformations used by MatheMagic.

Each transformation takes a GSM8K expression such as ``"52-8=44"`` and a seed,
and returns the (possibly rewritten) left-hand side, the answer under the
transformed rules, and a short log describing the rule that was applied.

The behaviour here is kept identical to the code used to produce the paper's
results, including some quirks (noted inline), so that the published test sets
can be regenerated exactly from their seeds. If you change anything in this
file, re-run ``scripts/generate_benchmark.py --check`` to confirm the shipped
data still regenerates bit-for-bit.
"""

import ast
import math
import random
import re

# The nine transformations evaluated in the paper, grouped as in Table 1.
TRANSFORMATION_GROUPS = {
    "fundamental": ["swap_symbols"],
    "numeric": ["swap_digits", "reverse_place_values", "rotate_digits"],
    "operations": ["add_constant", "negate_result", "modular_arithmetic"],
    "advanced": ["trig_substitution", "change_base"],
}
TRANSFORMATIONS = [t for group in TRANSFORMATION_GROUPS.values() for t in group]

_SAFE_GLOBALS = {
    "__builtins__": {"abs": abs, "int": int, "float": float, "pow": pow, "round": round},
    "math": math,
}
_SAFE_GLOBALS.update(math.__dict__)

def safe_eval(expression):
    """Evaluate an arithmetic expression with only math functions available."""
    return eval(expression, dict(_SAFE_GLOBALS))


def _draw_parameters(rng):
    """Draw every transformation parameter from ``rng`` in a fixed order.

    All parameters are drawn for every call (not only the one that is used) so
    that a given seed always produces the same parameters.
    """
    return {
        "op_mapping": {op: rng.choice(["+", "-", "*", "/", "%", "//"]) for op in ["+", "-", "*", "/"]},
        "digit_swap": tuple(rng.sample(range(10), 2)),
        "base": rng.randint(2, 12),
        "constant": rng.choice([i for i in range(-9, 10) if i != 0]),
        "modulo": rng.choice([2, 3, 5, 7, 10]),
        "trig_func": rng.choice(["math.sin", "math.cos"]),
        "trig_angle": rng.choice([0, 30, 45, 60, 90, 180, 270, 360]),
        "rotation": rng.randint(2, 5),
    }


def transform_expression(expression, seed, transformation):
    """Apply one transformation to a GSM8K expression.

    Args:
        expression: An expression with a single ``=``, e.g. ``"52-8=44"``.
        seed: Seed that fixes the transformation's parameters. Examples and the
            question in one test item share a seed, so they share the rule.
        transformation: One of ``TRANSFORMATIONS``.

    Returns:
        ``(problem, answer, log)``. ``problem`` is the left-hand side shown to
        the model, ``answer`` the result under the transformed rules (``None``
        if it cannot be evaluated, e.g. division by zero) and ``log`` a list of
        strings describing the rule (used to build the deductive hint).
    """
    if transformation not in TRANSFORMATIONS:
        raise ValueError(f"Unknown transformation {transformation!r}; expected one of {TRANSFORMATIONS}")

    rng = random.Random(seed)
    sides = expression.split("=")
    if len(sides) != 2:
        return expression, None, ["Invalid expression format"]

    problem, _ = sides
    expr = problem.strip()
    params = _draw_parameters(rng)
    log = []
    result_transforms = []

    if transformation == "swap_symbols":
        expr, entry = apply_symbol_swap(expr, params["op_mapping"])
        log.append(entry)
    elif transformation == "swap_digits":
        d1, d2 = params["digit_swap"]
        digit_map = {str(d1): str(d2), str(d2): str(d1)}
        expr = apply_digit_swap(expr, digit_map)
        log.append(f"Swapped digits: exchanged {d1} and {d2}")
        result_transforms.append(lambda r, dm=digit_map: apply_digit_map_to_result(r, dm))
    elif transformation == "change_base":
        base = params["base"]
        log.append(f"Changed numeral system to base {base}")
        result_transforms.append(
            lambda r, b=base: float_to_base_n(r, b) if isinstance(r, float) else int_to_base_n(r, b)
        )
    elif transformation == "negate_result":
        expr = f"-({expr})"
        log.append("Negated the expression")
    elif transformation == "add_constant":
        c = params["constant"]
        expr = f"({expr})+{c}"
        log.append(f"Added constant: increased the expression by {c}")
    elif transformation == "modular_arithmetic":
        m = params["modulo"]
        expr = f"({expr})%{m}"
        log.append(f"Applied modulo: result modulo {m}")
    elif transformation == "reverse_place_values":
        expr = apply_place_value_reversal(expr)
        log.append("Reversed place values: e.g., 123 becomes 321")
        result_transforms.append(apply_place_value_reversal_to_result)
    elif transformation == "rotate_digits":
        rot = params["rotation"]
        expr = apply_digit_rotation(expr, rot)
        log.append(f"Rotated digits: by {rot} position(s)")
        result_transforms.append(lambda r, rot=rot: apply_digit_rotation_to_result(r, -rot))
    elif transformation == "trig_substitution":
        expr, detail = apply_trig_substitution(expr, params["trig_func"], params["trig_angle"])
        if detail:
            log.append(f"Trigonometric substitution: {detail}")

    try:
        answer = safe_eval(expr)
        for fn in result_transforms:
            answer = fn(answer)
        return problem, answer, log
    except ZeroDivisionError:
        return problem, None, log + ["Error: Division by zero"]
    except Exception as e:  # malformed expression after rewriting
        return problem, None, log + [f"Evaluation error: {e}"]


# --- Individual transformations --------------------------------------------

def apply_symbol_swap(expression, op_mapping):
    """Swap arithmetic operators according to ``op_mapping`` (e.g. '+' -> '*')."""
    entry = "Operator mapping: " + str(op_mapping)
    relevant = [op for op in op_mapping if op in expression]
    out = expression
    # Placeholders stop a replaced operator from being replaced again.
    for i, op in enumerate(relevant):
        out = out.replace(op, f"§OP{i}§")
    for i, op in enumerate(relevant):
        out = out.replace(f"§OP{i}§", op_mapping[op])
    return out, entry


def apply_digit_swap(expression, digit_map):
    """Exchange two digits everywhere in the expression (e.g. 2 <-> 8)."""
    out = expression
    for digit in digit_map:
        out = out.replace(digit, f"§DIGIT{digit}§")
    for digit in digit_map:
        out = out.replace(f"§DIGIT{digit}§", digit_map[digit])
    return out


def apply_place_value_reversal(expression):
    """Reverse the integer digits of each number (123 -> 321, 12.5 -> 21.5)."""
    def reverse(match):
        s = match.group(0)
        if "." in s:
            int_part, dec_part = s.split(".")
            return str(int(int_part[::-1])) + "." + dec_part
        return str(int(s[::-1]))
    return re.sub(r"\b\d+(\.\d+)?\b", reverse, expression)


def apply_digit_rotation(expression, rotation=1):
    """Rotate the integer digits of each number right by ``rotation``."""
    def rotate(match):
        s = match.group(0)
        if "." in s:
            int_part, dec_part = s.split(".")
            if len(int_part) <= 1:
                return s
            r = rotation % len(int_part)
            return int_part[-r:] + int_part[:-r] + "." + dec_part
        if len(s) <= 1:
            return s
        r = rotation % len(s)
        return s[-r:] + s[:-r]
    return re.sub(r"\b\d+(\.\d+)?\b", rotate, expression)


def apply_trig_substitution(expression, trig_func, angle):
    """Replace the first integer with ``sin``/``cos`` of a fixed angle.

    Note: ``str.replace`` substitutes every occurrence of that integer's text,
    including inside longer numbers (e.g. "3" in "35"). Items where this makes
    the expression invalid are discarded during generation. Kept as-is so the
    published test sets regenerate exactly.
    """
    match = re.search(r"\b\d+\b", expression)
    if not match:
        return expression, None
    number = match.group(0)
    out = expression.replace(number, f"{trig_func}({math.radians(angle)})")
    return out, f"replaced {number} with {trig_func}({angle}°)"


# --- Result post-processing -------------------------------------------------

DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def int_to_base_n(n, base):
    if n < 0:
        return "-" + int_to_base_n(-n, base)
    if n < base:
        return DIGITS[n]
    out = ""
    while n:
        out = DIGITS[n % base] + out
        n //= base
    return out


def float_to_base_n(n, base, precision=6):
    """Convert a float to base ``base``; always writes ``precision`` fractional digits."""
    if n < 0:
        return "-" + float_to_base_n(-n, base, precision)
    integer_part = int(n)
    frac = n - integer_part
    digits = ""
    for _ in range(precision):
        frac *= base
        d = int(frac)
        digits += DIGITS[d]
        frac -= d
    return f"{int_to_base_n(integer_part, base)}.{digits}" if digits else int_to_base_n(integer_part, base)


def coerce_number_str(s):
    """'12.0' -> 12, '12.5' -> 12.5, '12' -> 12."""
    if "." in s:
        int_part, dec_part = s.split(".", 1)
        if dec_part == "" or set(dec_part) == {"0"}:
            return int(int_part)
        return float(s)
    return int(s)


def apply_digit_map_to_result(result, digit_map):
    s = str(coerce_number_str(str(result)))
    for d in digit_map:
        s = s.replace(d, f"§TMP{d}§")
    for d, rep in digit_map.items():
        s = s.replace(f"§TMP{d}§", rep)
    return coerce_number_str(s)


def _split_sign(result):
    s = str(coerce_number_str(str(result)))
    negative = s.startswith("-")
    return negative, (s[1:] if negative else s)


def apply_place_value_reversal_to_result(result):
    negative, s = _split_sign(result)
    if s.startswith("0."):
        core = s
    else:
        s = s.lstrip("0") or "0"
        if "." in s:
            ip, dp = s.split(".", 1)
            core = ip[::-1] + "." + dp
        else:
            core = s[::-1]
    return coerce_number_str(("-" if negative else "") + core)


def apply_digit_rotation_to_result(result, rotation=1):
    negative, s = _split_sign(result)
    if s.startswith("0."):
        core = s
    else:
        s = s.lstrip("0") or "0"
        if "." in s:
            ip, dp = s.split(".", 1)
            r = rotation % len(ip) if len(ip) > 1 else 0
            core = (ip[-r:] + ip[:-r] if r else ip) + "." + dp
        else:
            r = rotation % len(s) if len(s) > 1 else 0
            core = s[-r:] + s[:-r] if r else s
    return coerce_number_str(("-" if negative else "") + core)


# --- Natural-language rule descriptions (deductive hints) -------------------

def transforms_to_prompt(log):
    """Turn a transformation log into the hint text used in the deductive setting.

    The wording (including spacing quirks) matches the prompts used in the paper.
    """
    sentences = []
    for entry in log:
        if entry.startswith("Operator mapping:"):
            try:
                mapping = ast.literal_eval(entry[len("Operator mapping:"):].strip())
            except Exception:
                mapping = {}
            parts = [f"operator '{a}' is transformed into '{b}'" for a, b in mapping.items()]
            sentences.append("The operator mapping was applied, where " + " and ".join(parts) + ".")
        elif entry.startswith("Swapped digits:"):
            parts = entry.split("exchanged")
            if len(parts) == 2:
                sentences.append(f"The digits {parts[1].strip()} were exchanged.")
            else:
                sentences.append(f" {entry.capitalize()}.")
        elif entry.startswith("Changed numeral system"):
            sentences.append(f"The numeral system was changed to base {entry.split('base')[-1].strip()}.")
        elif entry.startswith("Negated"):
            sentences.append("The entire expression was negated.")
        elif entry.startswith("Added constant:"):
            c = entry.split("increased the expression by")[-1].strip()
            sentences.append(f"A constant value of {c} was added to the expression.")
        elif entry.startswith("Applied modulo:"):
            m = entry.split("result modulo")[-1].strip()
            sentences.append(f"Modulo arithmetic was applied with modulo {m}.")
        elif entry.startswith("Reversed place values:"):
            sentences.append(" the place values of digits were reversed (e.g., 123 becomes 321).")
        elif entry.startswith("Rotated digits:"):
            r = entry.split("by")[-1].split("position")[0].strip()
            sentences.append(f"Each number had its digits rotated by {r} position(s), excluding decimal parts.")
        elif entry.startswith("Trigonometric substitution:"):
            detail = entry[len("Trigonometric substitution:"):].strip()
            sentences.append(f"Trigonometric substitution was applied: {detail}.")
        else:
            sentences.append(f" {entry.capitalize()}.")

    has_operator_mapping = any("Operator mapping:" in e for e in log)
    has_digit_transform = any(
        key in e
        for e in log
        for key in ("Swapped digits", "Reversed digits", "Reversed place values", "Rotated digits", "Changed numeral system")
    )
    if has_digit_transform:
        if has_operator_mapping:
            sentences.append("Note: The operator mapping changes the meaning of arithmetic symbols, "
                             "while digit transformations adjust the representation of numbers.")
        else:
            sentences.append("Note: Digit transformations modify the representation of numbers, "
                             "including the final computed result.")
    return " ".join(sentences)
