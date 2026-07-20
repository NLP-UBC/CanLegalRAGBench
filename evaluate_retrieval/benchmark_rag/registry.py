"""
Component registry — resolve config strings like "embedders.hf.QwenEmbedder"
to the actual class and instantiate it with a config dict.

Usage
-----
    from benchmark_rag.registry import build

    embedder = build("embedders.hf.QwenEmbedder", {"model_name": "Qwen/Qwen3-Embedding-8B"})
    chunker  = build("chunkers.semantic.SemanticChunker", {"threshold": 0.6})

The string is resolved relative to `benchmark_rag.components`, so the caller
never has to manage imports.
"""

from __future__ import annotations

import importlib
from typing import Any


_COMPONENTS_PACKAGE = "benchmark_rag.components"


def build(type_path: str, config: dict[str, Any] | None = None) -> Any:
    """
    Instantiate a component from a dotted type path and an optional config dict.

    Parameters
    ----------
    type_path:
        Dotted path relative to `benchmark_rag.components`, e.g.
        ``"embedders.hf.QwenEmbedder"`` or a fully qualified path like
        ``"benchmark_rag.components.embedders.hf.QwenEmbedder"``.
    config:
        Keyword arguments forwarded to the class constructor.

    Returns
    -------
    An instance of the resolved class.
    """
    config = config or {}

    # Allow both relative ("embedders.hf.QwenEmbedder") and absolute paths.
    if not type_path.startswith(_COMPONENTS_PACKAGE):
        full_path = f"{_COMPONENTS_PACKAGE}.{type_path}"
    else:
        full_path = type_path

    module_path, class_name = full_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Could not import module '{module_path}'. "
            f"Make sure the component file exists and all dependencies are installed.\n"
            f"Original error: {e}"
        ) from e

    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ImportError(
            f"Module '{module_path}' has no class '{class_name}'. "
            f"Available names: {[n for n in dir(module) if not n.startswith('_')]}"
        )

    return cls(**config)


def build_from_component_config(component_cfg: dict[str, Any]) -> Any:
    """
    Convenience wrapper for Pydantic-validated component sub-configs of the form::

        {"type": "embedders.hf.QwenEmbedder", "model_name": "...", ...}

    The ``type`` key is popped and used as the type path; everything else is
    forwarded as constructor kwargs.
    """
    cfg = dict(component_cfg)
    type_path = cfg.pop("type")
    return build(type_path, cfg)
