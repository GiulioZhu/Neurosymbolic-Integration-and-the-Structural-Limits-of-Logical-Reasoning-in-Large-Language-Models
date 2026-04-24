"""
logicnli_adapter.py
===================
Dataset adapter for LogicNLI.

LogicNLI is a 4-label NLI benchmark that adds a "Paradox" relation to the
standard True / False / Unknown triple. A Paradox arises when both the
hypothesis H and its negation ¬H are entailed by the premises P.

Key modifications vs FOLIO:
  - Prompt: ICL examples include at least one Paradox sample so the model
    learns to identify contradictory reasoning paths.
  - Solver: dual Z3 check — if P ⊨ H AND P ⊨ ¬H → return "Paradox".
  - Fields: reads `hypothesis` + `premise` (singular) rather than
    `conclusion` + `premises`.

Label mapping:
  LogicNLI raw labels → canonical
    "entailment"    → "True"
    "contradiction" → "False"
    "neutral"       → "Unknown"
    "paradox"       → "Paradox"
"""

import sys, os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_adapters.base import DatasetAdapter
from code4logic.prompts import create_logicnli_prompt


class LogicNLIAdapter(DatasetAdapter):

    # Label mapping specific to LogicNLI
    _LABEL_MAP = {
        "entailment": "True",
        "contradiction": "False",
        "neutral": "Unknown",
        "paradox": "Paradox",
        "true": "True",
        "false": "False",
        "unknown": "Unknown",
    }

    # ── Field Extraction ──────────────────────────────────────────────────────

    def get_fields(self, row: dict) -> dict:
        hypothesis = str(row.get("hypothesis", "") or "").strip()
        premise    = str(row.get("premise", "")    or "").strip()
        raw_label  = str(row.get("label", "unknown")).strip().lower()
        label      = self._LABEL_MAP.get(raw_label, "Unknown")

        return {
            "nl_text":      hypothesis,
            "ground_truth": "",           # LogicNLI has no pre-annotated FOL
            "label":        label,
            "premises":     [premise] if premise else [],
            "context":      premise,
        }

    # ── Prompt Building ───────────────────────────────────────────────────────

    def get_prompt(self, row: dict, k_shots_df: pd.DataFrame,
                   num_examples: int = 10) -> str:
        fields = self.get_fields(row)
        return create_logicnli_prompt(
            hypothesis=fields["nl_text"],
            premise=fields["context"],
            k_shots_df=k_shots_df,
            num_examples=num_examples,
        )

    # ── Solver Label Post-processing (Paradox dual-check) ────────────────────

    def get_solver_label(self, z3_locals: dict, row: dict) -> str:
        """
        Extends the base 3-label check with Paradox detection.

        If the Z3 solver finds that BOTH:
          - P ∧ ¬H is unsatisfiable (P entails H), AND
          - P ∧  H is unsatisfiable (P entails ¬H)
        then the premises are internally contradictory → label as "Paradox".
        """
        res_ent  = str(z3_locals.get("res_ent",  "unknown"))
        res_cont = str(z3_locals.get("res_cont", "unknown"))

        if res_ent == "unsat" and res_cont == "unsat":
            return "Paradox"
        elif res_ent == "unsat" and res_cont == "sat":
            return "True"
        elif res_cont == "unsat" and res_ent == "sat":
            return "False"
        else:
            return "Unknown"

    def normalise_gold_label(self, raw_label: str) -> str:
        return self._LABEL_MAP.get(
            str(raw_label).strip().lower(), "Unknown"
        )
