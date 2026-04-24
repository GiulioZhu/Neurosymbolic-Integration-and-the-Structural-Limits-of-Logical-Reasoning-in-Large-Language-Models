"""
folio_adapter.py
================
Dataset adapter for FOLIO.

FOLIO features expert-written, multi-sentence natural language stories.
Key challenges:
  - Symbol drift: models may use different predicate names for the same concept
    across sentences (e.g., LivesIn vs ResidesIn).
  - Complex discourse context: premises are long, multi-sentence paragraphs.

Mitigation:
  - Global Signature Prompt: before ICL examples, inject a block that lists all
    predicate signatures seen in the k-shot examples to anchor the model's vocabulary.
  - Standard 3-label Z3 solver (True / False / Unknown).
"""

import sys, os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_adapters.base import DatasetAdapter
from code4logic.prompts import create_folio_prompt


class FolioAdapter(DatasetAdapter):

    # ── Field Extraction ──────────────────────────────────────────────────────

    def get_fields(self, row: dict) -> dict:
        conclusion = str(row.get("conclusion", "") or "").strip()
        premises_raw = row.get("premises", "")

        if isinstance(premises_raw, list):
            premises = [p.strip() for p in premises_raw if p.strip()]
        else:
            premises = [p.strip() for p in str(premises_raw).split("\n") if p.strip()]

        # nl_text is the conclusion sentence
        nl_text = conclusion
        if not nl_text and premises:
            nl_text = premises[0]

        return {
            "nl_text":      nl_text,
            "ground_truth": str(row.get("conclusion-FOL", "") or "").strip(),
            "label":        self.normalise_gold_label(str(row.get("label", "Unknown"))),
            "premises":     premises,
            "context":      "\n".join(premises),
        }

    # ── Prompt Building ───────────────────────────────────────────────────────

    def get_prompt(self, row: dict, k_shots_df: pd.DataFrame,
                   num_examples: int = 10) -> str:
        fields = self.get_fields(row)
        return create_folio_prompt(fields["nl_text"], k_shots_df, num_examples)
