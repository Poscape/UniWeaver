"""The release path must not import or launch the ablation layer."""
from __future__ import annotations

import re

PRODUCTION_DIRS = ("core", "baselines", "evaluation", "scripts",
                   "sft_prepare_data", "sft_train", "modeling", "data")

ABLATION_IMPORT = re.compile(r"^\s*(?:from|import)\s+ablation\b", re.MULTILINE)
ABLATION_ENTRY = re.compile(r"ablation/(?:run_variant|preflight)\.py")


def mentions_ablation(text: str) -> bool:
    """True when ``text`` imports the package or runs one of its entry points."""
    return bool(ABLATION_IMPORT.search(text) or ABLATION_ENTRY.search(text))
