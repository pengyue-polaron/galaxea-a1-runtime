"""Bind the released Euler student to immutable foundation components."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotPolicyServerConfig


def prepare_optional_imports() -> None:
    """Guard upstream's eager FlashAttention import in this torch-only adapter.

    The released requirements omit FlashAttention, but model.py imports its
    symbol even when attn_mode is torch. This process-local import shim rejects
    any accidental invocation; inference continues to use upstream torch SDPA.
    """
    if "flash_attn_interface" in sys.modules:
        return
    if (
        importlib.util.find_spec("flash_attn_interface") is not None
        or importlib.util.find_spec("flash_attn") is not None
    ):
        return
    module = ModuleType("flash_attn_interface")

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "Diffusion2One A1 adapter supports only tracked torch attention"
        )

    module.flash_attn_func = unavailable
    sys.modules[module.__name__] = module


def student_job(*, dtype: Any) -> SimpleNamespace:
    """Construct the released algorithm without importing environment overrides.

    This checkpoint is the Euler video/action student, not the unrelated
    non-diffusion action head or consistency sampler in the upstream repository.
    Sampling counts, guidance, normalization and dimensions are assigned from
    the validated deployment by the shared server adapter.
    """
    return SimpleNamespace(
        param_dtype=dtype,
        patch_size=(1, 2, 2),
        env_type="none",
        video_exec_step=-1,
        non_diffusion_action=False,
        consistency_video=False,
        mean_video_k=0,
        video_zero_noise=False,
        video_noise_seed=-1,
        action_noise_table="",
    )


def bind_foundation_loaders(module: Any, policy: LingBotPolicyServerConfig) -> None:
    for name, component in (
        ("load_vae", "vae"),
        ("load_text_encoder", "text_encoder"),
        ("load_tokenizer", "tokenizer"),
    ):
        original = getattr(module, name)
        path = str(policy.foundation_root / component)

        def load(
            _path: Any,
            *args: Any,
            _original=original,
            _path_override=path,
            **kwargs: Any,
        ) -> Any:
            return _original(_path_override, *args, **kwargs)

        setattr(module, name, load)
