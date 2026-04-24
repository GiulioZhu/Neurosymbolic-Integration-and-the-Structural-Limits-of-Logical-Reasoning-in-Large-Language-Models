"""
evaluate_nl2logic.py
====================
Bin-stratified evaluation of the NL2LOGIC pipeline.

- Custom Pipeline: Built specifically for this project to integrate the NL2LOGIC 
  pipeline with automated Z3 evaluation, bin-stratification, and dataset adapters.
"""

import os
import sys
import json
import argparse
import logging
import asyncio
import time
from tqdm import tqdm

# ── Add nl2logic + adapter to path ───────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_NL2DIR = os.path.join(_HERE, "nl2logic")
sys.path.insert(0, _NL2DIR)
sys.path.insert(0, _HERE)

from pipeline import Pipeline        # noqa: E402
from dataset_adapters import get_adapter  # noqa: E402

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────

def load_binned_dataset(path: str):
    """Load folio_binned.jsonl and group rows by complexity_bin."""
    bins: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # Only use validation samples for evaluation
            if row.get("split") != "validation":
                continue
            b = str(row.get("complexity_bin", "Unknown"))
            bins.setdefault(b, []).append(row)
    return bins


async def run_nl2logic_on_sample(pipeline: Pipeline, row: dict, sem: asyncio.Semaphore,
                                  adapter=None) -> dict:
    """
    Run the NL2LOGIC pipeline on one dataset row.
    Uses the dataset adapter for field extraction and Paradox-aware solver labelling.
    """
    async with sem:
        import time
        t_start = time.time()
        # ── Field Extraction via Adapter ──────────────────────────────────────
        if adapter is not None:
            fields     = adapter.get_fields(row)
            premises   = fields["premises"]
            conclusion = fields["nl_text"]
            gold_label = fields["label"]
        else:
            premises_raw = row.get("premises", "")
            premises = [p.strip() for p in str(premises_raw).split("\n") if p.strip()]
            conclusion = str(row.get("conclusion", ""))
            raw_label  = str(row.get("label", "Unknown"))
            gold_label = {"true": "True", "false": "False", "entailment": "True",
                          "contradiction": "False"}.get(raw_label.lower(), "Unknown")

        generated = "Error"
        prediction = "Error"
        match = False
        method = "error"
        reason = ""

        try:
            premise_asts = []
            for p in premises:
                ast = await pipeline.rephrase_and_parse(str(p), row=row)
                if ast is not None:
                    premise_asts.append(ast)
            
            hyp_ast = await pipeline.rephrase_and_parse(conclusion, row=row)
            if not hyp_ast:
                raise ValueError("Failed to parse hypothesis.")
                
            # ── Normalization Phase ──────────────────────────────────────────
            all_asts = []
            for ast in premise_asts:
                if hasattr(ast, "sentences"): all_asts.extend(ast.sentences)
                else: all_asts.append(ast)
            
            if hasattr(hyp_ast, "sentences"): all_asts.extend(hyp_ast.sentences)
            else: all_asts.append(hyp_ast)
            
            from ast_rl import RelationalLogic
            combined_logic = RelationalLogic(original_sentence="Combined Problem", sentences=all_asts)
            await pipeline.normalize_pipeline(combined_logic)
            
            # Collect Z3 declarations globally
            declarations = set()
            for s in all_asts:
                declarations.add(s.z3declaration_pass())
                    
            z3_code = "from z3 import *\nEntity = DeclareSort('Entity')\n"
            z3_code += "".join(sorted(list(declarations)))
            
            p_exprs = []
            for ast in premise_asts:
                if hasattr(ast, "sentences"):
                    for s in ast.sentences:
                        p_exprs.append(s.z3expression_pass())
                else:
                    p_exprs.append(ast.z3expression_pass())
                        
            h_exprs = []
            if hasattr(hyp_ast, "sentences"):
                for s in hyp_ast.sentences:
                    h_exprs.append(s.z3expression_pass())
            else:
                h_exprs.append(hyp_ast.z3expression_pass())
            
            if not h_exprs:
                raise ValueError("Empty hypothesis expressions.")
                
            h_expr = f"And({','.join(h_exprs)})" if len(h_exprs) > 1 else h_exprs[0]
            
            z3_code += f"\nP_exprs = [{','.join(p_exprs)}]\n"
            z3_code += f"H_expr = {h_expr}\n"
            
            z3_code += "\ns_ent = Solver()\n"
            z3_code += "s_ent.add(*P_exprs)\n"
            z3_code += "s_ent.add(Not(H_expr))\n"
            z3_code += "res_ent = s_ent.check()\n\n"
            
            z3_code += "s_cont = Solver()\n"
            z3_code += "s_cont.add(*P_exprs)\n"
            z3_code += "s_cont.add(H_expr)\n"
            z3_code += "res_cont = s_cont.check()\n\n"
            
            local_vars = {}
            try:
                exec(z3_code, {}, local_vars)
            except Exception as e:
                result_dir = os.path.join(_HERE, "..", "Result")
                with open(os.path.join(result_dir, f"failed_z3_debug_{row.get('example_id', 'unknown')}.py"), "w") as f_dbg:
                    f_dbg.write(z3_code)
                raise e

            # ── Adapter-aware solver labelling (Paradox support for LogicNLI) ─
            if adapter is not None:
                prediction = adapter.get_solver_label(local_vars, row)
            else:
                res_ent  = str(local_vars.get("res_ent",  ""))
                res_cont_v = str(local_vars.get("res_cont", ""))
                if res_ent == "unsat" and res_cont_v == "sat":
                    prediction = "True"
                elif res_cont_v == "unsat" and res_ent == "sat":
                    prediction = "False"
                else:
                    prediction = "Unknown"
            
            match = (prediction == gold_label)
            method = "z3_entailment"
            generated = f"Z3 Code Generated ({len(premise_asts)} premises mapped)"
            
        except Exception as exc:
            prediction = f"ERROR: {exc}"
            match = False
            method = "error"
            reason = str(exc)
            generated = prediction

        return {
            "nl":             conclusion,
            "generated_fol":  prediction,
            "ground_truth":   gold_label,
            "match":          match,
            "method":         method,
            "reason":         reason,
            "duration":       time.time() - t_start
        }


async def evaluate(backend: str, model: str, url: str, num_samples: int | None, verbose: bool, concurrency: int, dataset: str = "folio"):
    # ── Adapter ──────────────────────────────────────────────────────────────
    adapter = get_adapter(dataset)

    # ── Paths ────────────────────────────────────────────────────────────────
    result_root = os.path.join(_HERE, "..", "Result")
    ds_upper = dataset.upper()
    result_dir = os.path.join(result_root, ds_upper, "eval")
    os.makedirs(result_dir, exist_ok=True)
    
    filename = f"{dataset}_binned.jsonl"
    # Prioritise the global result dir for binned files
    binned_path = os.path.join(result_root, ds_upper, "data", filename)
    
    out_path    = os.path.join(result_dir, "evaluation_results_nl2logic.txt")
    raw_path    = os.path.join(result_dir, "nl2logic_raw_outputs.jsonl")

    if not os.path.exists(binned_path):
        print(f"[ERROR] Binned dataset [{dataset}] not found at {binned_path}.")
        sys.exit(1)

    bins = load_binned_dataset(binned_path)
    bin_keys = sorted(bins.keys())
    print(f"Loaded {sum(len(v) for v in bins.values())} validation samples across {len(bin_keys)} bins.")

    # ── Initialise pipeline ──────────────────────────────────────────────────
    print(f"\nInitialising NL2LOGIC pipeline — dataset={dataset!r}, model={model!r}")
    kwargs = {"llm": backend, "model": model, "logging": verbose}
    if backend == "vllm":
        kwargs["url"] = url
    pipeline = Pipeline(**kwargs)

    per_bin_results: dict[str, dict] = {}
    sem = asyncio.Semaphore(concurrency)

    with open(out_path, "w", encoding="utf-8") as f_out, \
         open(raw_path, "w", encoding="utf-8") as f_raw:

        f_out.write(f"NL2LOGIC Evaluation — dataset={dataset}, model={model}\n")
        f_out.write("=" * 60 + "\n\n")

        for b in bin_keys:
            rows = bins[b]
            if num_samples:
                rows = rows[:num_samples]

            correct = 0
            errors  = 0

            print(f"  [{b}] Starting batch of {len(rows)} samples...")
            start_time = time.time()
            
            tasks = [run_nl2logic_on_sample(pipeline, r, sem, adapter) for r in rows]
            
            with tqdm(total=len(rows), desc=f"  {b}", unit="sample", leave=False) as pbar:
                for i, coro in enumerate(asyncio.as_completed(tasks)):
                    res = await coro
                    
                    # ── Save Step-by-Step ──────────────────────────────────────────
                    f_raw.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f_raw.flush()

                    if res["method"] == "error":
                        errors += 1
                    elif res["match"]:
                        correct += 1

                    if verbose:
                        print(f"    [{b}] sample {i+1} completed in {res.get('duration', 0):.1f}s")

                    f_out.write(f"[{b}] Result {i+1}:\n")
                    f_out.write(f"  NL:          {res['nl']}\n")
                    f_out.write(f"  Generated:   {res['generated_fol']}\n")
                    f_out.write(f"  Ground Truth:{res['ground_truth']}\n")
                    f_out.write(f"  Match: {res['match']}  |  Method: {res['method']}\n")
                    f_out.write("-" * 40 + "\n")
                    f_out.flush()
                    
                    pbar.update(1)
            
            elapsed = time.time() - start_time
            print(f"  [{b}] Completed in {elapsed:.2f}s ({len(rows)/elapsed:.1f} samples/s)")

            total   = len(rows)
            acc     = correct / total * 100 if total > 0 else 0.0
            per_bin_results[b] = {
                "total": total, "correct": correct,
                "errors": errors, "accuracy": acc
            }

            summary = f"{b}: {acc:.1f}%  ({correct}/{total}, {errors} errors)"
            print(f"  {summary}          ")
            f_out.write(f"\n{summary}\n{'='*60}\n\n")

        # ── Final summary ────────────────────────────────────────────────────
        print("\n" + "=" * 50)
        print("Per-Bin Accuracy Summary (NL2LOGIC):")
        print("-" * 50)
        total_correct = 0
        total_all     = 0
        for b in bin_keys:
            r = per_bin_results[b]
            print(f"  {b}: {r['accuracy']:.1f}%  ({r['correct']}/{r['total']})")
            total_correct += r["correct"]
            total_all     += r["total"]
        overall = total_correct / total_all * 100 if total_all > 0 else 0
        print(f"\n  Overall: {overall:.1f}%  ({total_correct}/{total_all})")
        print("=" * 50)

        f_out.write(f"\nOverall Accuracy: {overall:.1f}%  ({total_correct}/{total_all})\n")

    print(f"\nDetailed results -> {out_path}")
    print(f"Raw outputs      -> {raw_path}")

    # Persist per-bin results for the plotting script
    per_bin_path = os.path.join(result_dir, "nl2logic_per_bin.json")
    with open(per_bin_path, "w", encoding="utf-8") as f:
        json.dump(per_bin_results, f, indent=2)
    print(f"Per-bin JSON     -> {per_bin_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NL2LOGIC on FOLIO (bin-stratified)")
    parser.add_argument("--backend", default="vllm",
                        choices=["vllm", "openai", "mock"],
                        help="LLM backend (default: vllm)")
    parser.add_argument("--model",   default="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                        help="Model name/path (default: Qwen/Qwen2.5-Coder-7B-Instruct-AWQ)")
    parser.add_argument("--url",     default="http://localhost:8000/v1",
                        help="vLLM server URL (only used with --backend vllm)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Max samples per bin (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable NL2LOGIC AST parse logging")
    parser.add_argument("--concurrency", type=int, default=128,
                        help="Async concurrency limit (default: 128)")
    parser.add_argument("--dataset", type=str, default="folio",
                        help="Dataset name (folio, malls, logicnli, nsa_lr)")
    args = parser.parse_args()
    
    asyncio.run(evaluate(
        args.backend, args.model, args.url, args.samples, args.verbose, args.concurrency, args.dataset
    ))


if __name__ == "__main__":
    main()
