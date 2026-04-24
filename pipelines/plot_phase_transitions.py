"""
Reads qwen_per_bin.json and nl2logic_per_bin.json from Result/ and
produces an Accuracy vs. LoCM Bin comparison plot.

Usage:
    python plot_phase_transitions.py
"""

import os
import json
import sys
import argparse

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
except ImportError:
    print("[ERROR] matplotlib is not installed. Run: pip install matplotlib")
    sys.exit(1)

_HERE      = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.path.join(_HERE, "..", "Result")

RANDOM_BASELINE = 1 / 3  # ~33.3% for True/False/Uncertain

def _load(res_dir: str, fname: str) -> dict | None:
    path = os.path.join(res_dir, fname)
    if not os.path.exists(path):
        print(f"[WARN] {fname} not found in {res_dir} — pipeline not yet evaluated.")
        return None
    with open(path) as f:
        return json.load(f)

def plot(dataset: str):
    ds_upper = dataset.upper()
    res_dir = os.path.join(RESULT_ROOT, ds_upper, "eval")
    viz_dir = os.path.join(RESULT_ROOT, ds_upper, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    qwen    = _load(res_dir, "qwen_per_bin.json")
    nl2     = _load(res_dir, "nl2logic_per_bin.json")

    if qwen is None and nl2 is None:
        print(f"[ERROR] No result files found for {dataset}. Run evaluation scripts first.")
        return

    # ── Determine common bin axis ──────────────────────────────────────────────
    all_bins = sorted(
        set(list(qwen.keys() if qwen else []) + list(nl2.keys() if nl2 else [])),
        key=lambda b: int(b.split()[-1]) if b.split()[-1].isdigit() else 99
    )
    x = np.arange(len(all_bins))

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    colour_qwen  = "#58a6ff"
    colour_nl2   = "#f78166"
    colour_base  = "#8b949e"

    def _acc(d: dict | None, bin_key: str) -> float | None:
        if d is None or bin_key not in d:
            return None
        return d[bin_key]["accuracy"]

    if qwen:
        y_qwen = [_acc(qwen, b) for b in all_bins]
        ax.plot(x, y_qwen, "-o", color=colour_qwen, linewidth=2.5,
                markersize=7, label="CODE4LOGIC (Qwen2.5-Coder-7B-Instruct-AWQ)", zorder=3)

    if nl2:
        y_nl2 = [_acc(nl2, b) for b in all_bins]
        ax.plot(x, y_nl2, "-s", color=colour_nl2, linewidth=2.5,
                markersize=7, label="NL2LOGIC (AST-guided)", zorder=3)

    # Random-guess baseline
    ax.axhline(
        y=RANDOM_BASELINE * 100, color=colour_base,
        linestyle="--", linewidth=1.5, label=f"Random baseline (~{RANDOM_BASELINE*100:.0f}%)"
    )

    # ── Styling ────────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(all_bins, color="white", fontsize=10)
    ax.set_xlabel("LoCM Complexity Bin", color="white", fontsize=12, labelpad=8)
    ax.set_ylabel("Accuracy (%)", color="white", fontsize=12, labelpad=8)
    ax.set_title(
        f"LPT Analytics [{dataset}]: Accuracy vs. LoCM Complexity",
        color="white", fontsize=14, pad=14
    )
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(axis="y", color="#30363d", linestyle="--", linewidth=0.8)

    legend = ax.legend(facecolor="#161b22", edgecolor="#30363d",
                       labelcolor="white", fontsize=10, loc="upper right")

    plt.tight_layout()

    out_path = os.path.join(viz_dir, "phase_transition_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot LPT analytics for a dataset.")
    parser.add_argument("--dataset", type=str, default="folio", help="Dataset name.")
    args = parser.parse_args()
    plot(args.dataset)
