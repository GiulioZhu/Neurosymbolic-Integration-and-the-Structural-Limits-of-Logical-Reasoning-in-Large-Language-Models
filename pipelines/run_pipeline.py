"""
run_pipeline.py
===============
Python orchestration script to run the full CODE4LOGIC evaluation pipeline locally.
This acts as a cross-platform "shell script".

Any arguments passed to this script will be forwarded to the evaluation scripts.
Example: python run_pipeline.py --samples 50 --backend vllm
"""

import os
import sys
import subprocess

def run_command(cmd_list, description):
    print(f"\n{'='*80}\n[RUNNING] {description}\n{'='*80}")
    print(f"Command: {' '.join(cmd_list)}")
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Step '{description}' failed with exit code {e.returncode}.")
        sys.exit(e.returncode)

def main():
    forward_args = sys.argv[1:]
    
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(os.path.dirname(here), "data", "FOLIO")
    binned_path = os.path.join(data_dir, "folio_binned.jsonl")

    # 1. Prepare dataset (takes no args)
    if not os.path.exists(binned_path):
        run_command(
            [sys.executable, os.path.join(os.path.dirname(here), "data", "preprocess.py")],
            "Step 1a: Downloading FOLIO dataset"
        )
        run_command(
            [sys.executable, os.path.join(os.path.dirname(here), "data", "prepare_dataset.py")],
            "Step 1b: Preparing FOLIO dataset & computing LoCM bins"
        )
    else:
        print(f"\n{'='*80}\n[SKIPPING] Step 1: Preprocessed dataset already exists at {binned_path}\n{'='*80}")

    # 2. Evaluate NL2LOGIC
    run_command(
        [sys.executable, os.path.join(here, "evaluate_nl2logic.py")] + forward_args,
        "Step 2: Evaluating NL2LOGIC pipeline"
    )

    # 3. Evaluate Qwen
    run_command(
        [sys.executable, os.path.join(here, "evaluate_qwen.py")] + forward_args,
        "Step 3: Evaluating CODE4LOGIC (Qwen) pipeline"
    )

    # 4. Plot results (takes no args)
    run_command(
        [sys.executable, os.path.join(here, "plot_phase_transitions.py")],
        "Step 4: Plotting phase transitions (Accuracy vs LoCM)"
    )

    print(f"\n{'='*80}\n[SUCCESS] Pipeline completed successfully!\n{'='*80}")

if __name__ == "__main__":
    main()
