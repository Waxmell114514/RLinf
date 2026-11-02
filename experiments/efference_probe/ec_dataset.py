"""X6-EC training samples from the training collection (PREREG_X6.md section 3.2).

One sample per logged call of a ``fault.mode: draw`` collection:

- **input:** the main frame, the task description, the state, and the
  efference input rebuilt by replaying the episode through the same
  :class:`ec_policy.EfferenceHistory` the harness uses online, so training
  and evaluation inputs are built by one piece of code;
- **target:** the policy's own 56 action tokens, with the dims the episode's
  fault changes replaced by the tokens of the oracle's command (the clean
  target is the policy's own output, token for token);
- **forward-model target** (arms D, E): the call's scaled state change, and
  the sent chunk it follows from.

Pure numpy and pandas; the model enters only through :class:`ActionTokenizer`'s
four numbers per dim, read from it by :meth:`ActionTokenizer.from_model`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from . import ec_policy
from .faults import parse as parse_fault

CHUNK, ACTION_DIM, STATE_DIM = 8, 7, 8
COLUMNS = [
    "episode_id",
    "call_idx",
    "task_id",
    "fault",
    "compensated",
    "a_cmd_env",
    "a_sent",
    "a_target",
    "action_tokens",
    "states_before",
    "states_after",
]
# Gate on the tokenizer: decoding the policy's own tokens must give back the
# command it logged.  One bin is 2/255 of the normalised range.
ROUND_TRIP_ATOL = 1e-3


@dataclass(frozen=True)
class ArmSpec:
    kind: str  # "copy" | "state"
    shuffle_history: bool
    forward_model: bool


ARMS = {
    "A": ArmSpec("copy", False, False),
    "B": ArmSpec("state", False, False),
    "C": ArmSpec("copy", True, False),
    "D": ArmSpec("state", False, True),
    "E": ArmSpec("copy", False, True),
}


class ActionTokenizer:
    """The rlinf OFT class's action decoding, and its inverse.

    Decoding (``predict_action_batch``): bin ``k = vocab - token - 1`` (clipped
    to the centres), centre ``bin_centers[k]``, then the BOUNDS_Q99
    unnormalisation on the masked dims.  Encoding takes the nearest centre.
    """

    def __init__(
        self,
        vocab_size: int,
        bin_centers: np.ndarray,
        low: Sequence[float],
        high: Sequence[float],
        mask: Optional[Sequence[bool]] = None,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.bin_centers = np.asarray(bin_centers, dtype=np.float64)
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.mask = (
            np.ones_like(self.low, dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
        )

    @classmethod
    def from_model(cls, model, unnorm_key: str) -> "ActionTokenizer":
        stats = model.get_action_stats(unnorm_key)
        return cls(
            model.vocab_size,
            model.bin_centers,
            stats["q01"],
            stats["q99"],
            stats.get("mask"),
        )

    def values(self, tokens: np.ndarray) -> np.ndarray:
        bins = np.clip(
            self.vocab_size - np.asarray(tokens, dtype=np.int64) - 1,
            0,
            len(self.bin_centers) - 1,
        )
        normalized = self.bin_centers[bins]
        scale = self.high - self.low + 1e-8
        return np.where(
            self.mask, 0.5 * (normalized + 1) * scale + self.low, normalized
        )

    def tokens(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        scale = self.high - self.low + 1e-8
        normalized = np.where(
            self.mask, np.clip(2 * (values - self.low) / scale - 1, -1, 1), values
        )
        nearest = np.abs(normalized[..., None] - self.bin_centers).argmin(axis=-1)
        return self.vocab_size - 1 - nearest


def target_tokens(
    own_tokens: np.ndarray,
    a_target: np.ndarray,
    fault_spec: str,
    tokenizer: ActionTokenizer,
) -> np.ndarray:
    """The policy's tokens, with the fault's dims set to the oracle's command."""
    out = np.asarray(own_tokens, dtype=np.int64).reshape(CHUNK, ACTION_DIM).copy()
    dims = list(parse_fault(fault_spec).dims)
    if dims:
        encoded = tokenizer.tokens(np.asarray(a_target).reshape(CHUNK, ACTION_DIM))
        out[:, dims] = encoded[:, dims]
    return out


# -- loading --------------------------------------------------------------------


def load_calls(run_dirs: Sequence[Path]) -> pd.DataFrame:
    """The calls of one or more collections, with their episodes' fields.

    ``episode`` is unique across runs; ``frame`` is the main JPEG's path.
    """
    frames = []
    for run_index, run_dir in enumerate(map(Path, run_dirs)):
        calls = pd.read_parquet(run_dir / "calls.parquet", columns=COLUMNS)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        episodes = pd.DataFrame(manifest["episodes"]).set_index("episode_id")
        if "task_description" not in episodes:
            raise ValueError(f"{run_dir} is not a training collection (no prompts)")
        calls["task_description"] = calls["episode_id"].map(
            episodes["task_description"]
        )
        calls["episode"] = run_index * 1_000_000 + calls["episode_id"]
        calls["frame"] = [
            str(run_dir / "frames" / f"ep{e:05d}" / f"call{c:04d}_main.jpg")
            for e, c in zip(calls["episode_id"], calls["call_idx"])
        ]
        frames.append(calls)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["episode", "call_idx"], kind="stable").reset_index(
        drop=True
    )


def _stack(calls: pd.DataFrame, column: str, *shape: int) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32) for v in calls[column]]).reshape(
        len(calls), *shape
    )


def fit_input_stats(
    calls: pd.DataFrame, proprio_bounds: Optional[tuple] = None
) -> dict[str, list[float]]:
    """Input normalisation, fixed once per collection and saved with each arm.

    State bounds: the checkpoint's proprio q01/q99 when it has them, else the
    collection's own.  Delta scale: the per-dim sd of the state change over
    the clean episodes.  Degenerate dims are widened so nothing divides by 0.
    """
    before = _stack(calls, "states_before", STATE_DIM)
    after = _stack(calls, "states_after", STATE_DIM)
    if proprio_bounds is not None:
        low, high = (np.asarray(b, dtype=np.float64) for b in proprio_bounds)
        source = "checkpoint"
    else:
        low = np.quantile(before, 0.01, axis=0)
        high = np.quantile(before, 0.99, axis=0)
        source = "collection"
    high = np.maximum(high, low + 1e-3)
    clean = (calls["fault"] == "clean").to_numpy()
    if not clean.any():
        raise ValueError("no clean episodes to scale the state changes by")
    scale = np.maximum((after - before)[clean].std(axis=0), 1e-4)
    return {
        "state_low": [float(v) for v in low],
        "state_high": [float(v) for v in high],
        "delta_scale": [float(v) for v in scale],
        "state_bounds_source": source,
    }


@dataclass
class Samples:
    efference: np.ndarray  # [N, in_dim]
    targets: np.ndarray  # [N, chunk, action_dim] token ids
    delta_target: np.ndarray  # [N, state_dim], scaled and clipped
    a_sent: np.ndarray  # [N, chunk * action_dim]
    states: np.ndarray  # [N, state_dim], raw
    episode: np.ndarray  # [N]
    frames: list[str]
    descriptions: list[str]

    def __len__(self) -> int:
        return len(self.episode)


def replay_history(
    calls: pd.DataFrame, kind: str, stats: dict[str, Any], history: int = 2
) -> np.ndarray:
    """Each call's efference input, as the harness builds it online.

    ``calls`` must be sorted by (episode, call_idx) with no call missing.
    """
    before = _stack(calls, "states_before", STATE_DIM)
    after = _stack(calls, "states_after", STATE_DIM)
    sent = _stack(calls, "a_sent", CHUNK, ACTION_DIM)
    builder = ec_policy.EfferenceHistory(
        1,
        kind,
        stats["state_low"],
        stats["state_high"],
        stats.get("delta_scale"),
        history=history,
        chunk=CHUNK,
        action_dim=ACTION_DIM,
    )
    efference = np.zeros((len(calls), builder.dim), np.float32)
    episodes = calls["episode"].to_numpy()
    call_index = calls["call_idx"].to_numpy()
    expected = 0
    for i in range(len(calls)):
        if i == 0 or episodes[i] != episodes[i - 1]:
            builder.reset()
            expected = 0
        if call_index[i] != expected:
            raise ValueError(f"episode {episodes[i]} skips call {expected}")
        expected += 1
        efference[i] = builder.vector(before[i : i + 1])[0]
        builder.push(sent[i : i + 1], before[i : i + 1], after[i : i + 1])
    return efference


def build_samples(
    calls: pd.DataFrame,
    kind: str,
    stats: dict[str, Any],
    tokenizer: ActionTokenizer,
    history: int = 2,
    own_target_calls: int = 0,
) -> Samples:
    """Replay every episode through the online input builder.

    ``own_target_calls`` (Amendment 4): on calls with ``call_idx`` below it,
    the target is the policy's own tokens on every dim, because the history
    there holds nothing yet that could tell a faulted episode from a clean
    one.  0 keeps Amendment 1's targets.
    """
    before = _stack(calls, "states_before", STATE_DIM)
    after = _stack(calls, "states_after", STATE_DIM)
    sent = _stack(calls, "a_sent", CHUNK, ACTION_DIM)
    efference = replay_history(calls, kind, stats, history)
    episodes = calls["episode"].to_numpy()
    targets = np.stack(
        [
            target_tokens(tokens, target, spec, tokenizer)
            for tokens, target, spec in zip(
                calls["action_tokens"], calls["a_target"], calls["fault"]
            )
        ]
    )
    early = calls["call_idx"].to_numpy() < own_target_calls
    if early.any():
        own = np.stack(
            [
                np.asarray(t, dtype=np.int64).reshape(CHUNK, ACTION_DIM)
                for t in calls.loc[early, "action_tokens"]
            ]
        )
        targets[early] = own
    scale = np.asarray(stats["delta_scale"], dtype=np.float32)
    delta = np.clip(
        (after - before) / scale, -ec_policy.DELTA_CLIP, ec_policy.DELTA_CLIP
    )
    return Samples(
        efference=efference,
        targets=targets,
        delta_target=delta.astype(np.float32),
        a_sent=sent.reshape(len(calls), -1),
        states=before,
        episode=episodes,
        frames=list(calls["frame"]),
        descriptions=list(calls["task_description"]),
    )


def round_trip_error(calls: pd.DataFrame, tokenizer: ActionTokenizer) -> float:
    """Max |decode(own tokens) - logged command| on dims 0-5 (the gate)."""
    tokens = np.stack([np.asarray(t) for t in calls["action_tokens"]])
    tokens = tokens.reshape(len(calls), CHUNK, ACTION_DIM)
    command = _stack(calls, "a_cmd_env", CHUNK, ACTION_DIM)
    return float(np.abs(tokenizer.values(tokens)[..., :6] - command[..., :6]).max())


def shuffled_history(
    samples: Samples, rng: np.random.Generator, state_dim: int = STATE_DIM
) -> np.ndarray:
    """Arm C: each sample's history blocks from a sample of another episode."""
    n = len(samples)
    donors = rng.permutation(n)
    clash = np.flatnonzero(samples.episode[donors] == samples.episode)
    for _ in range(100):
        if not len(clash):
            break
        donors[clash] = rng.integers(0, n, size=len(clash))
        clash = clash[samples.episode[donors[clash]] == samples.episode[clash]]
    if len(clash):
        raise ValueError("could not draw a history from another episode")
    out = samples.efference.copy()
    out[:, state_dim:] = samples.efference[donors, state_dim:]
    return out


def load_frames(paths: Sequence[str]) -> np.ndarray:
    """Main frames as uint8 [B, H, W, 3], as the env handed them to the model."""
    from PIL import Image

    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])


def family_counts(calls: pd.DataFrame) -> dict[str, int]:
    """Episodes per fault family and compensation, for the training log."""
    episodes = calls.groupby("episode").first()
    family = episodes["fault"].map(lambda spec: parse_fault(spec).family)
    key = family + np.where(episodes["compensated"], "+comp", "")
    return {str(k): int(v) for k, v in key.value_counts().sort_index().items()}
