"""
base.py
=======
Abstract base class for dataset-specific adapters.

Each adapter encapsulates three concerns:
  1. Field extraction  – get_fields(row)  →  standardised dict
  2. Prompt building   – get_prompt(...)  →  prompt string for the LLM
  3. Solver labelling  – get_solver_label(pred, row)  →  final string label
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class DatasetAdapter(ABC):
    """
    Base class for all dataset adapters.

    Subclasses must implement get_fields() and get_prompt().
    get_solver_label() has a sane default (3-label) that LogicNLI overrides.
    """

    # ── Field Extraction ──────────────────────────────────────────────────────

    @abstractmethod
    def get_fields(self, row: dict) -> dict:
        """
        Extract standardised fields from a raw dataset row.

        Returns a dict with the following keys (all str, never None):
            nl_text       – natural language text to parse / translate
            ground_truth  – gold FOL string (empty string if unavailable)
            label         – gold classification label (True/False/Unknown/Paradox)
            premises      – list[str] of NL premise sentences
            context       – full context / story string (may equal nl_text)
        """
        raise NotImplementedError

    # ── Prompt Building ───────────────────────────────────────────────────────

    @abstractmethod
    def get_prompt(self, row: dict, k_shots_df: pd.DataFrame,
                   num_examples: int = 10) -> str:
        """
        Construct the full ICL prompt for the CODE4LOGIC pipeline.

        Args:
            row          – the dataset row (already processed by get_fields)
            k_shots_df   – training-split DataFrame for ICL examples
            num_examples – number of ICL examples to include

        Returns:
            A complete prompt string ready to send to the LLM.
        """
        raise NotImplementedError

    # ── Solver Label Post-processing ─────────────────────────────────────────

    def get_solver_label(self, z3_locals: dict, row: dict) -> str:
        """
        Derive the final classification label from Z3 solver results.

        Default (FOLIO / NSA-LR): standard 3-label logic.
        LogicNLI overrides this to add Paradox detection.

        Args:
            z3_locals – dict of local variables after exec(z3_code, ...),
                        must contain 'res_ent' and 'res_cont'.
            row       – original dataset row (for context if needed).

        Returns:
            One of: "True", "False", "Unknown", or "Paradox".
        """
        from z3 import unsat, sat  # noqa: F401  (used via exec locals)
        res_ent  = z3_locals.get("res_ent")
        res_cont = z3_locals.get("res_cont")

        if res_ent is None or res_cont is None:
            return "Unknown"

        # Standard 3-label entailment check
        if str(res_ent) == "unsat" and str(res_cont) == "sat":
            return "True"
        elif str(res_cont) == "unsat" and str(res_ent) == "sat":
            return "False"
        else:
            return "Unknown"

    # ── Label Normalisation ───────────────────────────────────────────────────

    def normalise_gold_label(self, raw_label: str) -> str:
        """Normalise gold labels to the canonical set for this dataset."""
        mapping = {
            "true": "True", "entailment": "True",
            "false": "False", "contradiction": "False",
            "unknown": "Unknown", "neutral": "Unknown",
            "paradox": "Paradox",
        }
        return mapping.get(str(raw_label).strip().lower(), "Unknown")
