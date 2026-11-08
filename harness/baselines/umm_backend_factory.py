"""Composition root and metadata registry for unified UMM backends."""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendSpec:
    module: str
    loader: str
    adapter_source: str
    official_remote: str | None = None
    official_commit: str | None = None
    uniweaver_reported: bool = False


BACKEND_SPECS = {
    "bagel": BackendSpec(
        "baselines.bagel_umm_adapter",
        "load_bagel_backend",
        "baselines/bagel_umm_adapter.py",
        uniweaver_reported=True,
    ),
    "omnigen2": BackendSpec(
        "baselines.omnigen2_umm_adapter",
        "load_omnigen2_backend",
        "baselines/omnigen2_umm_adapter.py",
        "https://github.com/vectorspacelab/omnigen2",
        "18e6f9d5271b517fcb32e999f10df943ae9b8f20",
        uniweaver_reported=True,
    ),
    "show-o2": BackendSpec(
        "baselines.showo2_umm_adapter",
        "load_showo2_backend",
        "baselines/showo2_umm_adapter.py",
        "https://github.com/showlab/show-o",
        "45a5a2de01d1ebd10cd5864d29310a76476cdf23",
    ),
    "tokenflow": BackendSpec(
        "baselines.tokenflow_umm_adapter",
        "load_tokenflow_backend",
        "baselines/tokenflow_umm_adapter.py",
        "https://github.com/bytevisionlab/tokenflow",
        "9812da01c0695f4989060cc4eea053715e9e4441",
        uniweaver_reported=True,
    ),
    "illume-plus": BackendSpec(
        "baselines.illume_plus_umm_adapter",
        "load_illume_plus_backend",
        "baselines/illume_plus_umm_adapter.py",
        "https://github.com/illume-unified-mllm/illume_plus",
        "dfef7597f807df852a0ddbaf8fb2b33a423134ce",
    ),
    "seed-x": BackendSpec(
        "baselines.seedx_umm_adapter",
        "load_seedx_backend",
        "baselines/seedx_umm_adapter.py",
        "https://github.com/ailab-cvc/seed-x",
        "420d347dc50ba9ccdb8747efe9da3d250a50946c",
    ),
    "janus-pro": BackendSpec(
        "baselines.janus_pro_umm_adapter",
        "load_janus_pro_backend",
        "baselines/janus_pro_umm_adapter.py",
        "https://github.com/deepseek-ai/janus",
        "1daa72fa409002d40931bd7b36a9280362469ead",
    ),
}
SUPPORTED_BACKENDS = tuple(BACKEND_SPECS)


def backend_spec(name: str) -> BackendSpec:
    try:
        return BACKEND_SPECS[name]
    except KeyError as error:
        raise SystemExit(f"unsupported backend: {name}") from error


def load_backend(args):
    spec = backend_spec(args.backend)
    loader = getattr(importlib.import_module(spec.module), spec.loader)
    return loader(args)
