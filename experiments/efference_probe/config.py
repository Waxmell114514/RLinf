"""Configuration for the efference-probe collection runs.

Plain dataclasses plus a YAML loader.  Hydra is deliberately not used: the
harness is a standalone single-process script, so the extra composition
machinery would only obscure what is actually being run.  The resolved config
is hashed and written next to the data so every run is reproducible.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

# Hidden-state pools, defined on the *full* LM sequence.  See `capture.py` for
# the derivation; these names are used as keys throughout the analysis code.
POOL_NAMES = ("ctx_mean", "ctx_last", "act_mean")

TRANSFORMS = ("swap", "mirror", "freeze")


@dataclass
class ModelConfig:
    """Which checkpoint to probe and how to sample from it."""

    model_path: str = "/path/to/model/Openvla-oft-SFT-libero-goal-traj1/"
    model_type: str = "openvla_oft"
    implement_version: str = "rlinf"
    unnorm_key: str = "libero_goal_no_noops"
    precision: str = "bf16"
    action_dim: int = 7
    num_action_chunks: int = 8
    max_prompt_length: int = 128
    num_images_in_input: int = 1
    center_crop: bool = True
    attn_implementation: str = "flash_attention_2"
    # Sampling.  `do_sample: false` is greedy; see SPEC 3.7 for the smoke-test
    # rule that decides between greedy and the eval yaml's sampling params.
    do_sample: bool = False
    temperature: float = 1.6
    top_k: int = -1
    top_p: float = 1.0


@dataclass
class EnvConfig:
    """LIBERO suite selection and episode bookkeeping."""

    task_suite_name: str = "libero_goal"
    # Tasks to visit; null means "every task in the suite".
    task_ids: Optional[list[int]] = None
    # Number of fixed initial states (trials) per task.
    init_states_per_task: int = 20
    # Parallel env slots.  One batch holds `num_envs` init states of a *single*
    # task, so swap donors are always same-task (SPEC 3.1 T3).
    num_envs: int = 20
    seed: int = 0
    max_calls_per_episode: int = 40
    reset_gripper_open: bool = False
    camera_size: int = 256
    # Render only the last substep of each chunk.  Much faster; the cost is
    # that intermediate per-substep states are unavailable.
    skip_intermediate_renders: bool = True


@dataclass
class HijackConfig:
    """Hijack schedule (SPEC 3.2)."""

    enabled: bool = True
    transform: str = "swap"
    p_hijack: float = 0.25
    # No hijack before this call index (warm-up).
    warmup_calls: int = 4
    # Clean calls required after a hijack before the next one is allowed.
    # 2 => h_{m+1} reflects exactly one manipulated transition.
    min_clean_gap: int = 2
    # Draw swap donors from the same task only (always true with the
    # one-task-per-batch layout, kept explicit for auditability).
    swap_same_task_only: bool = True
    # Fraction of episodes flagged as probe episodes (SPEC 3.2).  The rest run
    # completely clean, giving E3 a within-run success-rate control instead of
    # a second collection pass.  Clean episodes still contribute negatives.
    probe_episode_fraction: float = 0.8


@dataclass
class CaptureConfig:
    """What to stash out of each forward pass."""

    layers: list[int] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 24, 28, 32])
    pools: list[str] = field(default_factory=lambda: list(POOL_NAMES))
    # P5 stretch: projector (vision -> LM) features.
    capture_vision: bool = False
    # One-time check that our sequence indexing reproduces the model's own
    # action logprobs.  Cheap; leave on.
    verify_indexing: bool = True


@dataclass
class FramesConfig:
    """Frame/video dumping."""

    # "none" | "episodes" (first `max_episodes`) | "all"
    mode: str = "episodes"
    max_episodes: int = 20
    jpeg_quality: int = 85
    save_wrist: bool = True


@dataclass
class BudgetConfig:
    """Hard stops (SPEC 0.5 / 4)."""

    max_gpu_hours: float = 4.0
    max_disk_gb: float = 30.0
    heartbeat_seconds: float = 300.0


@dataclass
class RunConfig:
    """Top-level collection config."""

    run_id: str = "dev"
    out_root: str = "experiments/efference_probe/data"
    notes: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    hijack: HijackConfig = field(default_factory=HijackConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    frames: FramesConfig = field(default_factory=FramesConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        """Stable short hash of the resolved config (excluding run_id/notes)."""
        payload = self.to_dict()
        payload.pop("run_id", None)
        payload.pop("notes", None)
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def validate(self) -> None:
        """Fail fast on configs that would silently produce unusable data."""
        if self.hijack.transform not in TRANSFORMS:
            raise ValueError(
                f"hijack.transform must be one of {TRANSFORMS}, "
                f"got {self.hijack.transform!r}"
            )
        if not 0.0 <= self.hijack.p_hijack <= 1.0:
            raise ValueError(f"hijack.p_hijack out of range: {self.hijack.p_hijack}")
        if not 0.0 <= self.hijack.probe_episode_fraction <= 1.0:
            raise ValueError(
                "hijack.probe_episode_fraction out of range: "
                f"{self.hijack.probe_episode_fraction}"
            )
        if self.hijack.min_clean_gap < 1:
            raise ValueError(
                "hijack.min_clean_gap must be >= 1 so that h_{m+1} follows a "
                "single manipulated transition"
            )
        if self.env.num_envs < 2 and self.hijack.transform == "swap":
            raise ValueError("swap hijacks need at least 2 parallel envs")
        if self.env.init_states_per_task < 1:
            raise ValueError("env.init_states_per_task must be >= 1")
        bad_pools = set(self.capture.pools) - set(POOL_NAMES)
        if bad_pools:
            raise ValueError(f"unknown capture.pools: {sorted(bad_pools)}")
        if not self.capture.layers:
            raise ValueError("capture.layers must not be empty")
        if len(set(self.capture.layers)) != len(self.capture.layers):
            raise ValueError("capture.layers contains duplicates")
        if self.frames.mode not in ("none", "episodes", "all"):
            raise ValueError(f"unknown frames.mode: {self.frames.mode!r}")
        if self.env.max_calls_per_episode <= self.hijack.warmup_calls + 1:
            raise ValueError(
                "env.max_calls_per_episode leaves no eligible calls after warm-up"
            )


def _build(cls, data: Any):
    """Recursively instantiate nested dataclasses from plain dicts."""
    if not dataclasses.is_dataclass(cls) or data is None:
        return data
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data)}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        field_type = fields[name].type
        # Nested dataclasses are the only compound types we build here.
        nested = {
            "ModelConfig": ModelConfig,
            "EnvConfig": EnvConfig,
            "HijackConfig": HijackConfig,
            "CaptureConfig": CaptureConfig,
            "FramesConfig": FramesConfig,
            "BudgetConfig": BudgetConfig,
        }.get(
            field_type
            if isinstance(field_type, str)
            else getattr(field_type, "__name__", "")
        )
        kwargs[name] = _build(nested, value) if nested else value
    return cls(**kwargs)


def load_config(path: str, overrides: Optional[list[str]] = None) -> RunConfig:
    """Load a YAML config and apply ``a.b=value`` overrides."""
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    cfg = _build(RunConfig, data)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must look like key.path=value: {override!r}")
        key, raw = override.split("=", 1)
        _apply_override(cfg, key.strip(), raw.strip())
    cfg.validate()
    return cfg


def _apply_override(cfg: RunConfig, key: str, raw: str) -> None:
    parts = key.split(".")
    target = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"unknown config path: {key}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise ValueError(f"unknown config path: {key}")
    # YAML parsing gives us ints/floats/bools/lists/None for free.
    setattr(target, leaf, yaml.safe_load(raw))
