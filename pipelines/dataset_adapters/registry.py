"""
registry.py
===========
Central registry mapping dataset names to their adapter classes.
"""

from dataset_adapters.folio_adapter    import FolioAdapter
from dataset_adapters.logicnli_adapter import LogicNLIAdapter
from dataset_adapters.nsa_lr_adapter   import NSALRAdapter

ADAPTERS = {
    "folio":    FolioAdapter,
    "logicnli": LogicNLIAdapter,
    "nsa_lr":   NSALRAdapter,
}


def get_adapter(dataset: str):
    """
    Return an instantiated adapter for the given dataset name.
    Raises ValueError for unknown datasets.
    """
    key = dataset.lower().strip()
    if key not in ADAPTERS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported: {list(ADAPTERS.keys())}"
        )
    return ADAPTERS[key]()
