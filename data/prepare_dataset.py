"""
Computes LoCM complexity scores and bins datasets into 9 complexity levels.
Standardizes field names across different datasets (FOLIO, LOGICNLI, NSA_LR, etc.)

Usage:
    python prepare_dataset.py --dataset folio
    python prepare_dataset.py --dataset logicnli
    python prepare_dataset.py --dataset nsa_lr
"""

import os
import json
import argparse
import pandas as pd
from locm_metric import compute_locm
from dataset_config import DATASET_CONFIG

_HERE = os.path.dirname(os.path.abspath(__file__))

def _load_jsonl(path: str, split_label: str) -> pd.DataFrame:
    """Load a newline-delimited JSON file into a DataFrame and tag the split."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["split"] = split_label
    return df

def main():
    parser = argparse.ArgumentParser(description="Compute LoCM scores and bin datasets.")
    parser.add_argument("--dataset", type=str, default="folio", choices=list(DATASET_CONFIG.keys()))
    args = parser.parse_args()

    config = DATASET_CONFIG[args.dataset]
    dataset_dir = os.path.join(_HERE, args.dataset.upper())
    
    # Map split paths
    train_path = os.path.join(dataset_dir, f"{args.dataset}_train.jsonl")
    val_path   = os.path.join(dataset_dir, f"{args.dataset}_validation.jsonl")

    print(f"Loading {args.dataset} dataset from local JSONL files ...")
    dataframes = []
    
    # Try loading train
    if os.path.exists(train_path):
        dataframes.append(_load_jsonl(train_path, "train"))
    
    # Try loading validation
    if os.path.exists(val_path):
        dataframes.append(_load_jsonl(val_path, "validation"))

    if not dataframes:
        print(f"[ERROR] No split files found for {args.dataset} in {dataset_dir}. Run preprocess.py first.")
        return

    df_eval = pd.concat(dataframes).reset_index(drop=True)
    print(f"Total samples to process: {len(df_eval)}")

    mapping = config["mapping"]
    locm_scores = []
    
    for idx, row in df_eval.iterrows():
        # 0. Check for pre-calculated score
        precalc_key = mapping.get("locm_score")
        if precalc_key and precalc_key in row:
            score = row[precalc_key]
        else:
            # 1. Hypothesis FOL
            hyp_key = mapping.get("hyp_fol")
            hyp_fol = str(row.get(hyp_key, "")) if hyp_key else ""

            # 2. Premises FOL
            prem_key = mapping.get("prem_fol")
            premises_raw = row.get(prem_key, [])
            if isinstance(premises_raw, str):
                prem_fol = premises_raw
            elif isinstance(premises_raw, list):
                prem_fol = " AND ".join([str(p) for p in premises_raw if p])
            else:
                prem_fol = ""

            # Construct full formula for score
            full_fol = (hyp_fol + " AND " + prem_fol).strip(" AND ")

            # 3. Hop proxy (premise count)
            prem_nl_key = mapping.get("prem_nl")
            premises_nl = row.get(prem_nl_key, [])
            if isinstance(premises_nl, list):
                num_premises = len(premises_nl)
            elif isinstance(premises_nl, str):
                # If it's a single string, try to split by '.' or just count as 1
                num_premises = len([p for p in premises_nl.split('.') if p.strip()]) 
            else:
                num_premises = 1 if premises_nl else 0

            score = compute_locm(full_fol, num_premises)
        
        locm_scores.append(score)

    df_eval["locm_score"] = locm_scores

    # Partition into 9 bins
    try:
        df_eval["complexity_bin"] = pd.qcut(
            df_eval["locm_score"].rank(method='first'), q=9, labels=[f"Bin {i}" for i in range(1, 10)]
        )
    except ValueError as e:
        print(f"Quantile binning failed ({e}), using equal-width binning instead.")
        df_eval["complexity_bin"] = pd.cut(
            df_eval["locm_score"], bins=9, labels=[f"Bin {i}" for i in range(1, 10)]
        )

    print("\nBin Distributions:")
    print(df_eval["complexity_bin"].value_counts().sort_index())

    # Save to Result/ and Dataset/ folders
    out_path = os.path.join(dataset_dir, config["binned_filename"])
    df_eval.to_json(out_path, orient="records", lines=True)
    
    # Also sync to Result/{DS}/data/ directory for the pipeline
    ds_upper = args.dataset.upper()
    result_dir = os.path.join(_HERE, "..", "Result", ds_upper, "data")
    os.makedirs(result_dir, exist_ok=True)
    
    global_res_path = os.path.join(result_dir, config["binned_filename"])
    df_eval.to_json(global_res_path, orient="records", lines=True)

    print(f"\nSaved binned dataset -> {out_path}")
    print(f"Also synced to experiment artifacts -> {global_res_path}")

if __name__ == "__main__":
    main()
