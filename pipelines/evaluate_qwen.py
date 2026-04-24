"""
evaluate_qwen.py
================
Bin-stratified evaluation of the CODE4LOGIC pipeline.

- Custom Pipeline: Built on top of the original repo's base components to handle 
  bin-stratification, local vLLM/Transformers execution, and custom dataset adapters.
"""

import os
import re
import sys
import json
import argparse
import pandas as pd
import torch
import warnings

# Add internal paths for local module resolution
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "code4logic"))
sys.path.insert(0, _HERE)

from transformers import AutoTokenizer, AutoModelForCausalLM
from basis_functions import *
from fol_verifier import check_equivalence
from dataset_adapters import get_adapter

warnings.filterwarnings("ignore")

# ── helpers ────────────────────────────────────────────────────────────────────

def _load_binned(data_dir: str, dataset_name: str) -> tuple[dict, pd.DataFrame]:
    """Load {dataset}_binned.jsonl → (bins_dict, train_df).
    bins_dict: bin_label -> list of row dicts (validation split only).
    """
    filename = f"{dataset_name}_binned.jsonl"
    ds_upper = dataset_name.upper()
    
    # Use absolute project root to avoid relative path confusion
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_root  = os.path.join(project_root, "Result")
    
    # Priority 1: Result/{DATASET}/data/
    path = os.path.join(result_root, ds_upper, "data", filename)
    
    if not os.path.exists(path):
        # Priority 2: data/{DATASET}/
        path = os.path.join(project_root, "data", ds_upper, filename)
        if not os.path.exists(path):
            # Fallback: Root Result folder (legacy)
            path = os.path.join(result_root, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Binned dataset [{dataset_name}] not found in any standard location."
                )
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df       = pd.DataFrame(rows)
    val_df   = df[df["split"] == "validation"].copy()
    train_df = df[df["split"] == "train"].copy()

    bins: dict[str, list[dict]] = {}
    for _, row in val_df.iterrows():
        b = str(row.get("complexity_bin", "Unknown"))
        bins.setdefault(b, []).append(row.to_dict())
    return bins, train_df


def _extract_fol_from_code(raw_code: str) -> str:
    """Run model output through code extraction + exec → FOL string."""
    if "natural_language_statement =" in raw_code:
        raw_code = raw_code.split("natural_language_statement =")[0]

    m = re.search(r'```(?:python)?\n(.*?)```', raw_code, re.DOTALL)
    if m:
        cleaned = m.group(1)
    else:
        m2 = re.search(r'```(?:python)?\n(.*)', raw_code, re.DOTALL)
        if m2:
            cleaned = m2.group(1)
        else:
            lines = raw_code.split('\n')
            cleaned = '\n'.join(
                l for l in lines
                if not l.strip().startswith('```')
                and any(kw in l for kw in ('formula', '=', 'End(', 'Predicate(', 'Variable('))
            )

    exec_globals = globals().copy()
    exec_locals: dict = {}
    try:
        exec(cleaned, exec_globals, exec_locals)
        return str(exec_locals.get('formula', 'Error'))
    except Exception as e:
        return f"Error: {e}"


def _generate_transformers(model, tokenizer, device, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kwargs: dict = {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    }
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def _generate_vllm(model_name: str, prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is not installed. Please pip install openai.")
    
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        top_p=0.9,
        max_tokens=1024,
        stop=["natural_language_statement", "===", "# Write", "```\n#", "Here is"]
    )
    return response.choices[0].message.content


# ── main evaluation ────────────────────────────────────────────────────────────

def evaluate(num_samples_per_bin: int | None = None, backend: str = "vllm", 
             model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ", dataset: str = "folio"):
    print("=" * 60)
    print(f"CODE4LOGIC Evaluation — {model_name} (bin-stratified)")
    print(f"Dataset: [{dataset}]")
    print("=" * 60)

    # Instantiate dataset-specific adapter
    adapter = get_adapter(dataset)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_root  = os.path.join(project_root, "Result")
    ds_upper     = dataset.upper()
    result_dir   = os.path.join(result_root, ds_upper, "eval")
    os.makedirs(result_dir, exist_ok=True)
    
    data_dir = os.path.join(project_root, "data", ds_upper)

    print(f"\nLoading binned {dataset} dataset ...")
    try:
        bins, train_df = _load_binned(data_dir, dataset)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    k_shots  = train_df.head(10)
    bin_keys = sorted(bins.keys())
    total_val = sum(len(v) for v in bins.values())
    print(f"  {total_val} validation samples across {len(bin_keys)} bins.")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    if backend == "transformers":
        print(f"\nLoading {model_name} on {device} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(device)
        model.eval()
    elif backend == "vllm":
        print(f"\nUsing vLLM backend with model: {model_name}")
        tokenizer, model = None, None
    else:
        raise ValueError(f"Unknown backend: {backend}")

    out_path = os.path.join(result_dir, "evaluation_results_qwen.txt")
    raw_path = os.path.join(result_dir, "qwen_raw_generations.txt")
    per_bin_results: dict[str, dict] = {}

    with open(out_path, "w", encoding="utf-8") as f_out, \
         open(raw_path, "w", encoding="utf-8") as raw_out:

        f_out.write(f"CODE4LOGIC ({model_name}) — Bin-Stratified Evaluation\n")
        f_out.write("=" * 60 + "\n\n")

        for b in bin_keys:
            rows = bins[b]
            if num_samples_per_bin:
                rows = rows[:num_samples_per_bin]

            correct = errors = 0

            for i, row in enumerate(rows):
                import time
                start_sample = time.time()
                
                print(f"  [{b}] sample {i+1}/{len(rows)} ...", end="\r")

                # Use adapter for dataset-specific field extraction
                fields       = adapter.get_fields(row)
                nl_text      = fields["nl_text"]
                ground_truth = fields["ground_truth"]

                try:
                    # Use adapter for dataset-specific prompt construction
                    prompt = adapter.get_prompt(row, k_shots, num_examples=10)
                    t0 = time.time()
                    if backend == "transformers":
                        raw_code = _generate_transformers(model, tokenizer, device, prompt)
                    elif backend == "vllm":
                        raw_code = _generate_vllm(model_name, prompt)
                    gen_time = time.time() - t0
                    
                    raw_out.write(f"=== [{b}] Sample {i+1} ===\n{raw_code}\n\n")
                    
                    t1 = time.time()
                    generated = _extract_fol_from_code(raw_code)
                    equiv     = check_equivalence(generated, ground_truth)
                    verify_time = time.time() - t1
                    
                    if equiv["match"]:
                        correct += 1
                    
                    if num_samples_per_bin is not None: # Proxy for verbose-like interest
                         print(f"  [{b}] sample {i+1} took {time.time()-start_sample:.1f}s (Gen: {gen_time:.1f}s, Eval: {verify_time:.1f}s)          ")
                except Exception as exc:
                    generated = f"Error: {exc}"
                    equiv     = {"match": False, "method": "error", "reason": str(exc)}
                    errors   += 1
                    print(f"\n      ⚠️  [{b}] sample {i+1} error: {exc}")

                f_out.write(f"[{b}] Sample {i+1}:\n")
                f_out.write(f"  NL:          {nl_text}\n")
                f_out.write(f"  Generated:   {generated}\n")
                f_out.write(f"  Ground Truth:{ground_truth}\n")
                f_out.write(f"  Match: {equiv['match']}  |  Method: {equiv['method']}\n")
                f_out.write("-" * 40 + "\n")

            total = len(rows)
            acc   = correct / total * 100 if total else 0.0
            per_bin_results[b] = {
                "total": total, "correct": correct, "errors": errors, "accuracy": acc
            }
            print(f"  [{b}] {acc:.1f}%  ({correct}/{total}, {errors} errors)          ")
            f_out.write(f"\n{b}: {acc:.1f}%  ({correct}/{total})\n{'=' * 60}\n\n")

        # ── Summary ────────────────────────────────────────────────────────────
        total_c = sum(r["correct"] for r in per_bin_results.values())
        total_a = sum(r["total"]   for r in per_bin_results.values())
        overall = total_c / total_a * 100 if total_a else 0.0
        print("\n" + "=" * 50)
        print("Per-Bin Accuracy Summary (CODE4LOGIC):")
        for b in bin_keys:
            r = per_bin_results[b]
            print(f"  {b}: {r['accuracy']:.1f}%  ({r['correct']}/{r['total']})")
        print(f"\n  Overall: {overall:.1f}%  ({total_c}/{total_a})")
        print("=" * 50)
        f_out.write(f"\nOverall: {overall:.1f}%  ({total_c}/{total_a})\n")

    per_bin_path = os.path.join(result_dir, "qwen_per_bin.json")
    with open(per_bin_path, "w") as f:
        json.dump(per_bin_results, f, indent=2)
    print(f"\nDetailed results → {out_path}")
    print(f"Per-bin JSON     → {per_bin_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CODE4LOGIC on FOLIO (bin-stratified)"
    )
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--samples", type=int, default=None,
                        help="Max samples per bin (default: all)")
    parser.add_argument("--backend", default="vllm",
                        choices=["transformers", "vllm"],
                        help="Backend to use for generation (default: vllm)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                        help="Model name for generation")
    parser.add_argument("--dataset", type=str, default="folio",
                        help="Dataset name (e.g., folio, malls, logicnli)")
    args = parser.parse_args()
    evaluate(num_samples_per_bin=args.samples, backend=args.backend, model_name=args.model, dataset=args.dataset)
