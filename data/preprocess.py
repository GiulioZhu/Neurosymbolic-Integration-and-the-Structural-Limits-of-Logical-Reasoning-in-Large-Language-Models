"""
Generalized downloader for reasoning datasets.
Uses dataset_config.py to map dataset names to HuggingFace or local paths.

Usage:
    python preprocess.py --dataset folio
    python preprocess.py --dataset logicnli
    python preprocess.py --dataset nsa_lr
"""

import os
import json
import argparse
from dataset_config import DATASET_CONFIG

_HERE = os.path.dirname(os.path.abspath(__file__))

def _save_split(records, path: str) -> None:
    """Write a list of dicts as newline-delimited JSON."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):>5} records -> {path}")

def _download_via_url(url: str, output_path: str) -> list:
    """Download a JSON file from a URL and return the contents."""
    import requests
    print(f"  Fetching from URL: {url} ...")
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "samples" in data:
        return data["samples"]
    else:
        return [data]

def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess datasets.")
    parser.add_argument("--dataset", type=str, default="folio", choices=list(DATASET_CONFIG.keys()),
                        help="Name of the dataset to process.")
    args = parser.parse_args()

    config = DATASET_CONFIG[args.dataset]
    output_dir = os.path.join(_HERE, args.dataset.upper())
    os.makedirs(output_dir, exist_ok=True)

    if config.get("custom_url"):
        print(f"Downloading {args.dataset} from custom URL...")
        records = _download_via_url(config["custom_url"], None)
        # For now, custom URLs are assumed to be the validation split
        out_path = os.path.join(output_dir, f"{args.dataset}_validation.jsonl")
        _save_split(records, out_path)
        print(f"\nDone. Run prepare_dataset.py --dataset {args.dataset} next.")
        return

    if config["hf_path"] is None:
        print(f"Dataset '{args.dataset}' is marked as LOCAL. Please ensure JSONL files are in {output_dir}")
        return

    print(f"Downloading {args.dataset} from HuggingFace ({config['hf_path']}) ...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("The 'datasets' package is required. Install with: pip install datasets")

    # We use trust_remote_code=True for legacy datasets that still require it
    dataset = load_dataset(config["hf_path"], trust_remote_code=True)

    for local_name, hf_name in config["splits"].items():
        if hf_name not in dataset:
            print(f"  [WARNING] Split '{hf_name}' not found in dataset - skipping.")
            continue
        
        out_path = os.path.join(output_dir, f"{args.dataset}_{local_name}.jsonl")
        records = [dict(row) for row in dataset[hf_name]]
        _save_split(records, out_path)

    print(f"\nDone. Run prepare_dataset.py --dataset {args.dataset} next to compute LoCM bins.")

if __name__ == "__main__":
    main()
