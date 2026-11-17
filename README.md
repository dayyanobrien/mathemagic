# MatheMagic: Generating Dynamic Mathematics Benchmarks Robust to Memorization
Paper: [arXiv:2510.05962](https://arxiv.org/abs/2510.05962)

MatheMagic changes the rules of arithmetic: `+` might mean multiplication, digits might be
swapped, every answer might be shifted by a constant. Expressions come from GSM8K, and every
answer can still be checked automatically. A model has to work out the new rule from
in-context examples (**inductive** setting) or apply a rule that is described to it
(**deductive** setting). Test sets are generated from a random seed, so you can reproduce the
paper's sets exactly or make fresh ones that no model has seen.

## Contents

```
mathemagic/       library: transformations, test-set generation, prompts, scoring
scripts/
  generate_benchmark.py      regenerate the paper's test sets, or make new ones from a seed
  evaluate.py                run a model (vLLM or any OpenAI-compatible API) and score it
  score.py                   score responses produced by your own inference code
  generate_finetune_data.py  build fine-tuning data
  generate_noise_ablation.py regenerate the noise-ablation set
  noise_ablation_report.py   compare a model on the noise ablation and the benchmark
verifier/                    alternative-explanation verifier 
data/
  benchmark/                   the paper's 5 test sets (seeds 42, 68, 100, 789, 1024)
  noise_ablation/              examples that follow no single rule (seed 42)
  finetune/                    fine-tuning splits used in the paper
```

## Install

```bash
pip install -r requirements.txt
pip install vllm          # only for local GPU inference
```

## The transformations

| Group | Transformation | Example rule |
|---|---|---|
| Fundamental | `swap_symbols` | `+` means `*`, `-` means `%`, ... |
| Numeric | `swap_digits` | the digits 2 and 8 are exchanged everywhere |
| | `reverse_place_values` | 123 is read as 321 |
| | `rotate_digits` | digits are rotated by *k* places |
| Operations | `add_constant` | every result is increased by *c* |
| | `negate_result` | every result is negated |
| | `modular_arithmetic` | every result is taken modulo *m* |
| Advanced | `trig_substitution` | the first number is replaced by sin/cos of an angle |
| | `change_base` | answers are written in base *b* (2-12) |

Each test set has 405 items: 45 per transformation. Each item has its own seed, so the
parameters (which operators, which constant, which base, ...) vary from item to item. An
item's in-context examples always follow the same rule as its question.

## Data format

`data/benchmark/seed_<seed>.jsonl` holds one item per line:

```json
{
  "id": "42-swap_symbols-0",
  "seed": 42,
  "transformation_type": "swap_symbols",
  "problem": "52-8",
  "answer": 4,
  "original": "52-8=44",
  "hint": "The operator mapping was applied, where operator '+' is transformed into '//' and ...",
  "examples": [{"problem": "14+7", "answer": 2}, {"problem": "50*6", "answer": 2}, "... 512 in total"]
}
```

* `problem` / `answer`: the question and its answer under the new rules. `change_base`
  answers are strings in that base, e.g. `"1A3"`.
* `original`: the GSM8K expression under ordinary maths. It is used to detect "reversion"
  (the model answering with ordinary maths).
* `hint`: the description of the rule shown in the deductive setting.
* `examples`: in-context examples. A *k*-shot prompt uses the first *k*.

Prompts are built with `mathemagic.build_messages(item, shots, setting, instruction_variant)`.
A 2-shot inductive prompt looks like this:

```
[system] You are a helpful AI assistant in a world where mathematical notation has changed. Use only
         the provided context. Always answer using decimal numbers only.
         Instruction:
         Present your solution in the following format: ... 4. Final Answer: LaTeX \boxed{answer}.
[user]   Examples:
         1. 14+7 = \boxed{2}
         2. 50*6 = \boxed{2}
         Problem:
         52-8 = \boxed{?}
```

In the deductive setting, `(Hint: <hint>)` is added on the line before the problem.

## Evaluating a model

The paper's main setup is greedy decoding, instruction variant 1, all 5 seeds, and
0-512 shots:

```bash
# Local model with vLLM
python scripts/evaluate.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct

# Any OpenAI-compatible API (OpenAI, `vllm serve`, Together, OpenRouter, ...)
export OPENAI_API_KEY=...
python scripts/evaluate.py --backend openai --model gpt-4o-mini --shots 0 8 64

# Self-hosted server
python scripts/evaluate.py --backend openai --base-url http://localhost:8000/v1 --model my-model

# Deductive setting, all four instruction variants
python scripts/evaluate.py --model ... --setting deductive --instruction-variant 1 2 3 4
```

Results are written to `results/<model>/<dataset>/`:

* `details.jsonl`: one row per prompt, with the raw output, the prediction and its score.
  Interrupted runs resume from this file.
* `summary.csv`: mean and standard deviation across seeds for each setting, instruction
  variant and shot count.
* `summary_per_seed.csv` and `summary_per_transformation.csv`.

Reported metrics, in %:

* `accuracy`: correct under the new rules. Answers count as correct within a relative and
  absolute tolerance of 1e-4.
* `correct_wrt_orig`: the answer is correct under ordinary maths (reversion).
* `no_or_empty_box`, `parse_errors`, `wrong_other`: the error breakdown.

### Using your own inference code

```bash
python scripts/evaluate.py --dump-prompts prompts.jsonl --shots 0 8 64
# Run the `messages` in each line through your model, then write
# {"key": <key>, "output": <text>} per line to responses.jsonl
python scripts/score.py responses.jsonl --model my-model
```

## Making new test sets

Because test sets are generated from a seed, you can make a fresh one that has never been
published:

```bash
python scripts/generate_benchmark.py --seeds 2027 --out-dir data/new
python scripts/evaluate.py --model ... --data data/new/benchmark
```

`python scripts/generate_benchmark.py --check` confirms that the shipped files regenerate
byte-for-byte. GSM8K is downloaded from the Hugging Face Hub (`openai/gsm8k`). Questions
come from the GSM8K test split, and in-context examples from the train split.

To add a transformation, add a branch to `transform_expression` and a hint sentence to
`transforms_to_prompt` in `mathemagic/transforms.py`.

## Verifier

Several rules can explain the same few examples, so a response can be "wrong" while still
following a valid rule. To check for this, the verifier:

1. Gives the model's reasoning to Gemini, which turns it into a Python `solver`.
2. Runs that solver on every example in the prompt and on the question.

If the solver reproduces every example and the model's own answer, the response counts as
correct. This only applies to inductive responses with at least one shot.
`trig_substitution` is excluded, as in the paper.

```bash
export GEMINI_API_KEY=...
python verifier/generate_solvers.py results/<model>/benchmark/details.jsonl   # Gemini Batch API
python verifier/check_solvers.py   results/<model>/benchmark/details.jsonl --solvers verifier_solvers.jsonl
```

`check_solvers.py` executes model-generated code. Run it in a sandbox if that is a concern.

## Fine-tuning

`data/finetune/` contains the 8-shot splits used in the paper. Each dataset has 2,500 train
and 625 test items, and the two splits share no expressions:

* `format_dataset_combined_*`: all nine transformations.
* `format_dataset_incl_<T>_*`: only transformation T.
* `format_dataset_excl_<T>_*`: every transformation except T (leave-one-out).

Each line has chat `messages` ready for supervised fine-tuning. It also has
`messages_original`, the same prompt under ordinary maths. To make new splits:

```bash
python scripts/generate_finetune_data.py --out-dir data/finetune_new --seed 0
```

## License

Code and data are released under the [MIT License](LICENSE). The base expressions come from
[GSM8K](https://github.com/openai/grade-school-math) (MIT License, Copyright (c) 2021 OpenAI).

## Citation

If you use MatheMagic, please cite:

```bibtex
@misc{obrien2025mathemagicgeneratingdynamicmathematics,
      title={MatheMagic: Generating Dynamic Mathematics Benchmarks Robust to Memorization}, 
      author={Dayyán O'Brien and Barry Haddow and Emily Allaway and Pinzhen Chen},
      year={2025},
      eprint={2510.05962},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.05962}, 
}
```
