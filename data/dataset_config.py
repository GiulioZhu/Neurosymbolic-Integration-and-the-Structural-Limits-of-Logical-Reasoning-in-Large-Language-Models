"""
Registry of supported datasets for the pipelines.
Defines metadata, field mappings, and splits for standardized processing.
"""

DATASET_CONFIG = {
    "folio": {
        "hf_path": "yale-nlp/FOLIO",
        "splits": {
            "train": "train",
            "validation": "validation"
        },
        "mapping": {
            "prem_fol": "premises-FOL",
            "hyp_fol": "conclusion-FOL",
            "prem_nl": "premises",
            "hyp_nl": "conclusion",
            "label": "label"
        },
        "label_space": ["True", "False", "Unknown"],
        "binned_filename": "folio_binned.jsonl"
    },
    "logicnli": {
        "hf_path": "tasksource/LogicNLI",
        "splits": {
            "train": "train",
            "validation": "test"
        },
        "mapping": {
            "prem_nl": "premise",
            "hyp_nl": "hypothesis",
            "label": "label"
        },
        "label_space": ["True", "False", "Unknown", "Paradox"],
        "binned_filename": "logicnli_binned.jsonl"
    },
    "nsa_lr": {
        "hf_path": None,
        "custom_url": "https://raw.githubusercontent.com/AI4SS/Logical-Phase-Transitions/main/dataset/NSA-LR/test/test.json",
        "splits": {
            "validation": "test.json"
        },
        "mapping": {
            "prem_nl": "context",
            "hyp_fol": "conclusion_fol",
            "label": "answer",
            "locm_score": "complexity"  # Pre-calculated — bypass LoCM recomputation
        },
        "label_space": ["True", "False", "Unknown"],
        "binned_filename": "nsa_lr_binned.jsonl"
    }
}
