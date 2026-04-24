"""
nsa_lr_adapter.py
=================
Dataset adapter for NSA-LR (Logical Phase Transitions benchmark).

NSA-LR is specifically designed for fine-grained LoCM analysis. Unlike FOLIO
and LogicNLI which focus on a single premise→conclusion translation, NSA-LR
provides exhaustive INTERMEDIATE reasoning steps as structured FOL chains.

Key differences from FOLIO:
  - Full chain translation: the model is prompted to translate every intermediate
    reasoning step, not just the final hypothesis.
  - Pre-calculated complexity: NSA-LR's `complexity` field directly encodes
    nesting depth (d) and premise count (N_φ), so no LoCM re-derivation needed.
  - Field mapping: `context` → premises, `answer` → label, `complexity` → locm_score.

Label mapping:
  NSA-LR answer values → canonical
    "A" or "true"        → "True"
    "B" or "false"       → "False"
    "C" or "uncertain"   → "Unknown"
"""

import sys, os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_adapters.base import DatasetAdapter
from code4logic.prompts import create_nsa_lr_prompt


class NSALRAdapter(DatasetAdapter):
    _LABEL_MAP = {
        "a": "True",  "true": "True",  "yes": "True",
        "b": "False", "false": "False", "no": "False",
        "c": "Unknown", "uncertain": "Unknown", "unknown": "Unknown",
    }

    # ── Field Extraction ──────────────────────────────────────────────────────

    def get_fields(self, row: dict) -> dict:
        context = str(row.get("context", "") or "").strip()
        raw_label = str(row.get("answer", "unknown")).strip().lower()
        label = self._LABEL_MAP.get(raw_label, "Unknown")

        # The conclusion_fol field contains a structured chain of FOL expressions
        # We use the final one as ground_truth and expose the full chain for chain-mode
        conclusion_fol = row.get("conclusion_fol", "")
        if isinstance(conclusion_fol, list):
            ground_truth = str(conclusion_fol[-1]) if conclusion_fol else ""
            fol_chain = conclusion_fol
        else:
            ground_truth = str(conclusion_fol or "").strip()
            fol_chain = [ground_truth] if ground_truth else []

        # Premises are encoded in the context (one per line or single string)
        premises = [p.strip() for p in context.split("\n") if p.strip()]

        return {
            "nl_text":      context,           # Full context as NL input
            "ground_truth": ground_truth,       # Final FOL in the chain
            "label":        label,
            "premises":     premises,
            "context":      context,
            "fol_chain":    fol_chain,          # Full intermediate FOL chain
            "locm_score":   row.get("complexity", 0),  # Pre-calculated complexity
        }

    # ── Prompt Building ───────────────────────────────────────────────────────

    def get_prompt(self, row: dict, k_shots_df: pd.DataFrame,
                   num_examples: int = 10) -> str:
        fields = self.get_fields(row)
        return create_nsa_lr_prompt(
            context=fields["context"],
            k_shots_df=k_shots_df,
            num_examples=num_examples,
        )

    def normalise_gold_label(self, raw_label: str) -> str:
        return self._LABEL_MAP.get(str(raw_label).strip().lower(), "Unknown")
