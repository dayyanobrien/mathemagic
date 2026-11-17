"""Evaluate a model on MatheMagic.

Backends
--------
``vllm``    Load a Hugging Face model locally with vLLM (greedy decoding), as in the paper.
``openai``  Any OpenAI-compatible chat API: OpenAI, a ``vllm serve`` endpoint,
            Together, OpenRouter, etc. Set ``--base-url`` and the API key env var.

Examples
--------
Paper setup for one open model (inductive, instruction variant 1, all shots):

    python scripts/evaluate.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct

Deductive setting via an OpenAI-compatible server, a few shot counts:

    python scripts/evaluate.py --backend openai --model gpt-4o-mini \\
        --setting deductive --shots 0 8 64

Noise ablation (examples that follow no single rule; 8 and 64 shots by default):

    python scripts/evaluate.py --backend vllm --model Qwen/Qwen2.5-14B-Instruct --data noise_ablation

Write prompts only (to run with your own inference code, then use score.py):

    python scripts/evaluate.py --dump-prompts prompts.jsonl --shots 0 8

Runs resume: rows already in the output ``details.jsonl`` are skipped.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mathemagic.benchmark import PAPER_SHOTS, SETTINGS  # noqa: E402
from mathemagic.noise_ablation import NOISE_SHOTS  # noqa: E402
from mathemagic.evaluation import (  # noqa: E402
    DATASETS, append_details, detail_row, iter_jobs, load_items, read_details, sanitize, write_summaries,
)


def merge_system_into_user(messages):
    """For chat templates without a system role."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if system and rest:
        rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
    return rest


# --- Backends ----------------------------------------------------------------

class VLLMBackend:
    def __init__(self, args):
        import torch
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size or max(torch.cuda.device_count(), 1),
            max_model_len=args.max_model_len,
            dtype="auto",
            trust_remote_code=args.trust_remote_code,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
        self.max_model_len = args.max_model_len
        self.chat_template_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else {}

    def generate(self, batch_messages):
        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                      **self.chat_template_kwargs)
                   for m in batch_messages]
        keep = [i for i, p in enumerate(prompts)
                if len(self.tokenizer.encode(p, add_special_tokens=False)) <= self.max_model_len]
        outputs = [None] * len(prompts)  # prompts that are too long get no output (scored as format errors)
        if keep:
            for i, out in zip(keep, self.llm.generate([prompts[i] for i in keep], self.params, use_tqdm=False)):
                outputs[i] = out.outputs[0].text
        return outputs


class OpenAIBackend:
    def __init__(self, args):
        import openai

        self.client = openai.OpenAI(base_url=args.base_url, api_key=os.environ.get(args.api_key_env, "EMPTY"))
        self.model = args.model
        self.workers = args.concurrency
        self.kwargs = {}
        if args.temperature is not None:
            self.kwargs["temperature"] = args.temperature
        if args.max_new_tokens:
            self.kwargs[args.max_tokens_field] = args.max_new_tokens
        if args.reasoning_effort:
            self.kwargs["reasoning_effort"] = args.reasoning_effort
        if args.extra_body:
            self.kwargs["extra_body"] = json.loads(args.extra_body)

    def _one(self, messages, attempts=5):
        for attempt in range(attempts):
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages, **self.kwargs)
                return resp.choices[0].message.content
            except Exception as e:  # rate limits, timeouts, ...
                if attempt == attempts - 1:
                    print(f"Request failed after {attempts} attempts: {e}", file=sys.stderr)
                    return None
                time.sleep(2 ** attempt)

    def generate(self, batch_messages):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self._one, batch_messages))


# --- Main ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="Model name (HF id for vllm, model id for the API).")
    p.add_argument("--backend", choices=["vllm", "openai"], default="vllm")
    p.add_argument("--data", nargs="+", default=["benchmark"],
                   help=f"Dataset name ({', '.join(DATASETS)}), directory or JSONL file(s).")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Default: all seeds in the data.")
    p.add_argument("--shots", type=int, nargs="+", default=None,
                   help="Default: 0-512 (benchmark) or 8 and 64 (noise_ablation).")
    p.add_argument("--setting", nargs="+", choices=SETTINGS, default=["inductive"])
    p.add_argument("--instruction-variant", type=int, nargs="+", choices=[1, 2, 3, 4], default=[1])
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--no-system-role", action="store_true", help="Merge the system prompt into the user turn.")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--batch-size", type=int, default=256, help="Prompts sent to the backend at once.")
    p.add_argument("--dump-prompts", metavar="PATH", help="Write prompts to PATH (JSONL) and exit.")
    vg = p.add_argument_group("vllm backend")
    vg.add_argument("--max-model-len", type=int, default=32768, help="Longer prompts are skipped.")
    vg.add_argument("--tensor-parallel-size", type=int, default=None, help="Default: all visible GPUs.")
    vg.add_argument("--trust-remote-code", action="store_true")
    vg.add_argument("--chat-template-kwargs", help='JSON, e.g. \'{"enable_thinking": false}\'')
    og = p.add_argument_group("openai backend")
    og.add_argument("--base-url", default=None, help="Default: https://api.openai.com/v1")
    og.add_argument("--api-key-env", default="OPENAI_API_KEY")
    og.add_argument("--concurrency", type=int, default=16)
    og.add_argument("--temperature", type=float, default=0.0, help="Use a negative value to omit it.")
    og.add_argument("--max-tokens-field", default="max_tokens",
                    help="Some APIs/models need max_completion_tokens instead.")
    og.add_argument("--reasoning-effort", default=None)
    og.add_argument("--extra-body", default=None, help="JSON merged into the request body.")
    args = p.parse_args()
    if args.temperature is not None and args.temperature < 0:
        args.temperature = None
    if not args.dump_prompts and not args.model:
        p.error("--model is required unless --dump-prompts is given")
    if args.shots is None:
        args.shots = NOISE_SHOTS if args.data == ["noise_ablation"] else PAPER_SHOTS
    return args


def main():
    args = parse_args()
    items = load_items(args.data, args.seeds)
    jobs = list(iter_jobs(items, args.setting, args.shots, args.instruction_variant))
    if args.no_system_role:
        for job in jobs:
            job["messages"] = merge_system_into_user(job["messages"])

    if args.dump_prompts:
        with open(args.dump_prompts, "w", encoding="utf-8") as f:
            for job in jobs:
                f.write(json.dumps({"key": job["key"], "id": job["item"]["id"], "seed": job["item"]["seed"],
                                    "setting": job["setting"], "instruction_variant": job["instruction_variant"],
                                    "shots": job["shots"], "messages": job["messages"]}) + "\n")
        print(f"Wrote {len(jobs)} prompts to {args.dump_prompts}. Score responses with scripts/score.py.")
        return

    dataset_label = "+".join(sanitize(os.path.basename(str(d).rstrip("/"))) for d in args.data)
    out_dir = os.path.join(args.out_dir, sanitize(args.model), dataset_label)
    details_path = os.path.join(out_dir, "details.jsonl")
    done = {row["key"] for row in read_details(details_path)}
    todo = [job for job in jobs if job["key"] not in done]
    print(f"{len(jobs)} prompts ({len(done)} already done, {len(todo)} to run) -> {details_path}")

    if todo:
        backend = VLLMBackend(args) if args.backend == "vllm" else OpenAIBackend(args)
        for start in tqdm(range(0, len(todo), args.batch_size), desc="batches"):
            batch = todo[start:start + args.batch_size]
            outputs = backend.generate([job["messages"] for job in batch])
            append_details([detail_row(job, args.model, out)
                            for job, out in zip(batch, outputs)], details_path)

    rows = read_details(details_path)
    summary = write_summaries(rows, out_dir)
    if summary is not None:
        cols = ["setting", "instruction_variant", "shots", "accuracy", "accuracy_std", "correct_wrt_orig"]
        if "refusal" in summary:  # noise ablation: there is no correct answer
            cols = ["setting", "shots", "correct_wrt_orig", "refusal", "wrong_other", "no_or_empty_box"]
        print(summary[cols].to_string(index=False, float_format="%.1f"))
        print(f"Summaries written to {out_dir}/summary*.csv")


if __name__ == "__main__":
    main()
