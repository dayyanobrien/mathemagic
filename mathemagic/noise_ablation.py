"""Noise ablation: in-context examples that follow no single rule.

Could models fail at induction simply because they treat counterfactual
examples as noise and fall back to ordinary maths? To test this, every
in-context example in this set applies a *different* transformation (with its
own random parameters), so no single rule explains the examples. If models
already treated the real benchmark's examples as noise, they would behave the
same way on both sets. Instead, models answer with ordinary arithmetic much
more often here than on the real task.

The set has 405 questions (seed 42) with 64 examples each; a k-shot prompt
uses the first k. Example transformations are shuffled in cycles of nine, so
the first nine examples use every transformation once and no transformation
appears twice in a row.

There is no correct answer, so responses are sorted into four categories:

* ``normal_math``  - the boxed answer equals the ordinary-arithmetic answer
* ``refusal``      - no numeric answer, and the response says the examples
                     are inconsistent, ambiguous, ...
* ``other_answer`` - any other number
* ``format_err``   - no boxed number at all

The transformation code below is the implementation used for this experiment
(it follows the fine-tuning data generator). It is kept as is, rather than
replaced by ``mathemagic.transforms``, so that the shipped set regenerates
exactly.
"""

import ast
import math
import random
import re

NOISE_SEED = 42
NOISE_QUERIES = 405
NOISE_MAX_EXAMPLES = 64
NOISE_SHOTS = [8, 64]  # shot counts reported in the paper

TRANSFORMS = (
    "swap_symbols",
    "swap_digits",
    "reverse_place_values",
    "rotate_digits",
    "add_constant",
    "negate_result",
    "modular_arithmetic",
    "trig_substitution",
    "change_base",
)

_REFUSAL_RE = re.compile(
    r"\b(?:cannot|can't|unable|impossible|inconsistent|contradict(?:ion|ory|s)?|"
    r"no (?:single|consistent|clear)|not enough information|insufficient information|"
    r"underdetermined|ambiguous|arbitrary|unclear)\b",
    re.IGNORECASE,
)
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
_NUMBER_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")


# --- Arithmetic -----------------------------------------------------------------

def safe_number(expression):
    """Evaluate the arithmetic subset used by the GSM8K expressions."""
    tree = ast.parse(expression.strip(), mode="eval")
    binary = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }
    unary = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            return unary[type(node.op)](visit(node.operand))
        raise ValueError(f"Unsupported arithmetic: {ast.dump(node)}")

    return visit(tree)


def normal_answer(expression):
    """The ordinary-arithmetic answer of ``"lhs=rhs"`` (or of a bare expression)."""
    lhs, separator, rhs = expression.partition("=")
    return coerce_number(safe_number(rhs if separator else lhs))


def coerce_number(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def numbers_equal(left, right):
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-4, abs_tol=1e-4)
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()


# --- Transformations ------------------------------------------------------------

def _int_to_base_n(number, base):
    if number == 0:
        return "0"
    if number < 0:
        return "-" + _int_to_base_n(-number, base)
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""
    while number:
        result = digits[number % base] + result
        number //= base
    return result


def _float_to_base_n(number, base, precision=6):
    if number < 0:
        return "-" + _float_to_base_n(-number, base, precision)
    integer_part = int(number)
    result = _int_to_base_n(integer_part, base)
    fractional_part = number - integer_part
    if fractional_part == 0:
        return result
    fractional_result = ""
    for _ in range(precision):
        fractional_part *= base
        digit = int(fractional_part)
        fractional_result += "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[digit]
        fractional_part -= digit
        if fractional_part == 0:
            break
    return f"{result}.{fractional_result}"


def _replace_with_placeholders(text, mapping, prefix):
    result = text
    for index, source in enumerate(mapping):
        # Letter-only sentinels cannot collide with digit or operator mappings.
        result = result.replace(source, f"§{prefix}{chr(65 + index)}§")
    for index, (source, target) in enumerate(mapping.items()):
        result = result.replace(f"§{prefix}{chr(65 + index)}§", target)
    return result


def _reverse_place_values(expression):
    def reverse(match):
        value = match.group(0)
        if "." in value:
            integer, decimal = value.split(".")
            return f"{int(integer[::-1])}.{decimal}"
        return str(int(value[::-1]))

    return re.sub(r"\b\d+(?:\.\d+)?\b", reverse, expression)


def _rotate_digits(expression, rotation):
    def rotate(match):
        value = match.group(0)
        integer, dot, decimal = value.partition(".")
        if len(integer) <= 1:
            return value
        amount = rotation % len(integer)
        rotated = integer[-amount:] + integer[:-amount]
        return rotated + (dot + decimal if dot else "")

    return re.sub(r"\b\d+(?:\.\d+)?\b", rotate, expression)


def _to_number_if_possible(text):
    try:
        return coerce_number(float(text))
    except ValueError:
        return text


def _transform_result_digits(value, mapping):
    return _to_number_if_possible(_replace_with_placeholders(str(coerce_number(value)), mapping, "DIGIT"))


def _reverse_result(value):
    text = str(coerce_number(value))
    negative = text.startswith("-")
    if negative:
        text = text[1:]
    integer, dot, decimal = text.partition(".")
    result = integer[::-1] + (dot + decimal if dot else "")
    return _to_number_if_possible(("-" if negative else "") + result)


def _rotate_result(value, rotation):
    text = str(coerce_number(value))
    negative = text.startswith("-")
    if negative:
        text = text[1:]
    integer, dot, decimal = text.partition(".")
    if len(integer) > 1:
        amount = rotation % len(integer)
        integer = integer[-amount:] + integer[:-amount]
    result = integer + (dot + decimal if dot else "")
    return _to_number_if_possible(("-" if negative else "") + result)


def transformed_answer(expression, transform, seed):
    """Apply ``transform`` (parameters drawn from ``seed``) to ``"lhs=rhs"``.

    Returns ``(lhs, answer)``: the problem as shown and its answer under the rule.
    """
    rng = random.Random(seed)
    lhs = expression.split("=", 1)[0].strip()
    op_mapping = {op: rng.choice(["+", "-", "*", "/", "%", "//"]) for op in "+-*/"}
    digit_pair = tuple(rng.sample(range(10), 2))
    base = rng.randint(2, 12)
    constant = rng.choice([number for number in range(-9, 10) if number])
    modulo = rng.choice([2, 3, 5, 7, 10])
    trig_func = rng.choice(["sin", "cos"])
    trig_angle = rng.choice([0, 30, 45, 60, 90, 180, 270, 360])
    rotation = rng.randint(2, 5)
    transformed = lhs

    def postprocess(value):
        return value

    if transform == "swap_symbols":
        transformed = _replace_with_placeholders(transformed, op_mapping, "OP")
    elif transform == "swap_digits":
        first, second = map(str, digit_pair)
        digit_mapping = {first: second, second: first}
        transformed = _replace_with_placeholders(transformed, digit_mapping, "DIGIT")
        def postprocess(value):
            return _transform_result_digits(value, digit_mapping)
    elif transform == "reverse_place_values":
        transformed = _reverse_place_values(transformed)
        postprocess = _reverse_result
    elif transform == "rotate_digits":
        transformed = _rotate_digits(transformed, rotation)
        def postprocess(value):
            return _rotate_result(value, -rotation)
    elif transform == "add_constant":
        transformed = f"({transformed})+{constant}"
    elif transform == "negate_result":
        transformed = f"-({transformed})"
    elif transform == "modular_arithmetic":
        transformed = f"({transformed})%{modulo}"
    elif transform == "trig_substitution":
        match = re.search(r"\b\d+(?:\.\d+)?\b", transformed)
        if match:
            trig_value = getattr(math, trig_func)(math.radians(trig_angle))
            transformed = transformed[:match.start()] + repr(trig_value) + transformed[match.end():]
    elif transform == "change_base":
        def postprocess(value):
            return _float_to_base_n(value, base) if isinstance(value, float) else _int_to_base_n(value, base)
    else:
        raise ValueError(f"Unknown transform: {transform}")

    return lhs, postprocess(safe_number(transformed))


def _transformation_schedule(rng, count):
    """Shuffled cycles of all nine transformations, never repeating at a cycle boundary."""
    schedule = []
    while len(schedule) < count:
        cycle = list(TRANSFORMS)
        rng.shuffle(cycle)
        if schedule and cycle[0] == schedule[-1]:
            cycle[0], cycle[1] = cycle[1], cycle[0]
        schedule.extend(cycle)
    return schedule[:count]


# --- Test-set generation --------------------------------------------------------

def create_noise_set(test_expressions, train_expressions, seed=NOISE_SEED,
                     queries=NOISE_QUERIES, max_examples=NOISE_MAX_EXAMPLES):
    """Generate the noise-ablation set.

    Takes the expression lists from ``mathemagic.load_gsm8k_expressions()``.
    Items use the benchmark's format, so ``build_messages`` and the evaluation
    scripts work unchanged. ``answer`` is ``None`` (there is no correct
    answer), and each example records the transformation and seed that made it.
    """
    rng = random.Random(seed)
    train = list(train_expressions)
    test = list(test_expressions)
    rng.shuffle(train)
    rng.shuffle(test)
    cursor = 0
    items = []

    for index, query in enumerate(test[:queries]):
        schedule = _transformation_schedule(rng, max_examples)
        examples = []
        while len(examples) < max_examples:
            transform = schedule[len(examples)]
            expression = train[cursor % len(train)]
            cursor += 1
            if expression == query:
                continue
            example_seed = rng.randrange(2 ** 32)
            try:
                problem, answer = transformed_answer(expression, transform, example_seed)
                normal_answer(expression)
            except (ArithmeticError, OverflowError, SyntaxError, TypeError, ValueError):
                continue
            examples.append({"problem": problem, "answer": coerce_number(answer),
                             "transformation": transform, "seed": example_seed})

        items.append({
            "id": f"{seed}-mixed-{index}",
            "seed": seed,
            "transformation_type": "mixed",
            "problem": query.split("=", 1)[0].strip(),
            "answer": None,
            "original": query,
            "hint": "",
            "examples": examples,
        })
    return items


# --- Scoring --------------------------------------------------------------------

def score_noise_response(output, item):
    """Sort one response to a noise-ablation item into its category.

    Returns the same flags as ``mathemagic.score_response`` (``correct`` is
    always ``False``; ``correct_orig`` means ``normal_math``) plus ``refusal``.
    """
    boxed = _BOXED_RE.findall(output or "")
    prediction = boxed[-1].strip().replace(",", "") if boxed else ""
    value = None
    if _NUMBER_RE.fullmatch(prediction):
        try:
            value = float(prediction)
        except ValueError:
            value = None
    refusal_signal = bool(_REFUSAL_RE.search(output or ""))
    normal = value is not None and numbers_equal(value, normal_answer(item["original"]))
    return {
        "prediction": prediction,
        "correct": False,
        "correct_orig": normal,
        "format_err": value is None and not refusal_signal,
        "parse_err": False,
        "other_err": value is not None and not normal,
        "refusal": value is None and refusal_signal,
    }


def rule_overlaps(item):
    """Positions of the examples whose rule maps the *question* to its ordinary answer.

    For such a rule, following it and doing ordinary maths give the same answer.
    """
    ordinary = normal_answer(item["original"])
    hits = []
    for position, example in enumerate(item["examples"]):
        try:
            _, value = transformed_answer(item["original"], example["transformation"], example["seed"])
        except (ArithmeticError, OverflowError, SyntaxError, TypeError, ValueError):
            continue
        if numbers_equal(value, ordinary):
            hits.append(position)
    return hits


def matched_normal_math_rate(rows, items, shots, draws=400, rng_seed=0):
    """Ordinary-maths rate with the benchmark's overlap exclusion applied.

    On the benchmark, questions whose transformed answer equals the ordinary
    answer are excluded, since there the two cannot be told apart. A noise
    prompt shows many rules, but a model can follow only one, so the matched
    exclusion picks one of the shown rules at random per question and drops the
    question if that rule gives the ordinary answer. This is repeated ``draws``
    times. Returns ``(mean %, 2.5th percentile, 97.5th percentile, mean kept n)``.
    """
    by_id = {item["id"]: item for item in items}
    hits = {}
    for row in rows:
        if row["id"] not in hits:
            hits[row["id"]] = set(h for h in rule_overlaps(by_id[row["id"]]) if h < shots)
    rng = random.Random(rng_seed)
    estimates, kept = [], []
    for _ in range(draws):
        keep = [r for r in rows if rng.randrange(shots) not in hits[r["id"]]]
        if keep:
            estimates.append(100 * sum(r["correct_orig"] for r in keep) / len(keep))
            kept.append(len(keep))
    estimates.sort()
    return (sum(estimates) / len(estimates), estimates[int(0.025 * len(estimates))],
            estimates[int(0.975 * len(estimates))], sum(kept) / len(kept))
