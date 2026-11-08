from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _restore_variant_patches():
    patches = sys.modules.get("ablation.patches")
    if patches is None:
        yield
        return
    patches.restore()
    yield
    patches.restore()
