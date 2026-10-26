#!/usr/bin/env python
"""E4 checkpoint checks: how checkpoint A loads, and whether a local copy is it.

Two subcommands, neither needing torch or a GPU::

    # Is Base-Lora's root merged, or base weights waiting for lora_adapter/?
    python experiments/efference_probe/e4_ckpt_check.py merged-test

    # Is this local directory byte-for-byte the pinned revision?
    python experiments/efference_probe/e4_ckpt_check.py verify \
        --ckpt sft --path /ckpts/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

``merged-test`` reads single tensors out of the published safetensors shards
with HTTP range requests (tens of MB, not 15 GB) and compares Base-Lora's root
against ``openvla/openvla-7b`` (the base its ``adapter_config.json`` names),
its own ``lora_adapter/``, and the GRPO checkpoint.  The harness hard-codes
``is_lora=False``, so this is the check that decides whether A can be loaded
as-is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PLAN_PATH = Path(__file__).resolve().parent / "e4_plan.yaml"
HF = "https://huggingface.co"
BASE_REPO = "openvla/openvla-7b"

# One LoRA-targeted LLM weight, one LoRA-targeted vision weight, and one weight
# no adapter touches (a merged LoRA must leave it bit-identical to the base).
TARGETED = (
    "language_model.model.layers.0.self_attn.q_proj.weight",
    "vision_backbone.featurizer.blocks.0.attn.qkv.weight",
)
UNTARGETED = "language_model.model.layers.0.input_layernorm.weight"


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# -- remote safetensors access ---------------------------------------------


def _range(url: str, start: int, stop: int) -> bytes:
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{stop - 1}"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


class RemoteShards:
    """Read individual bf16 tensors from a repo's safetensors over HTTP."""

    def __init__(self, repo: str, revision: str = "main") -> None:
        self.base = f"{HF}/{repo}/resolve/{revision}"
        self._headers: dict[str, tuple[dict[str, Any], int]] = {}
        self._index: dict[str, str] | None = None

    def json(self, name: str) -> Any:
        with urllib.request.urlopen(f"{self.base}/{name}", timeout=120) as response:
            return json.load(response)

    def _header(self, url: str) -> tuple[dict[str, Any], int]:
        if url not in self._headers:
            size = struct.unpack("<Q", _range(url, 0, 8))[0]
            self._headers[url] = (json.loads(_range(url, 8, 8 + size)), 8 + size)
        return self._headers[url]

    def tensor(self, key: str, file: str | None = None) -> np.ndarray:
        if file is None:
            if self._index is None:
                self._index = self.json("model.safetensors.index.json")["weight_map"]
            file = self._index[key]
        url = f"{self.base}/{file}"
        header, data_start = self._header(url)
        meta = header[key]
        if meta["dtype"] != "BF16":
            raise TypeError(f"{key}: expected BF16, got {meta['dtype']}")
        start, stop = meta["data_offsets"]
        raw = np.frombuffer(
            _range(url, data_start + start, data_start + stop), np.uint16
        )
        # bf16 is the top half of a float32.
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


def effective_rank(delta: np.ndarray, stored: np.ndarray, margin: float = 3.0) -> dict:
    """Rank of ``delta`` above the noise floor bf16 storage puts under it.

    ``stored`` was rounded to bf16, so ``stored - base`` carries a full-rank
    rounding residue on top of any low-rank update; a relative cut-off would
    count that residue as rank.  The floor is the top singular value of a
    simulated rounding error (uniform within +-ulp/2 of each stored element).
    """
    ulp = 2.0 ** (np.floor(np.log2(np.abs(stored) + 1e-30)) - 7)
    noise = np.random.default_rng(0).uniform(-0.5, 0.5, size=stored.shape) * ulp
    floor = float(np.linalg.svd(noise, compute_uv=False)[0])
    singular = np.linalg.svd(delta, compute_uv=False)
    rank = int((singular > margin * floor).sum())
    energy = np.cumsum(singular**2) / np.sum(singular**2)
    return {
        "rank": rank,
        "noise_floor_sv": floor,
        "sv_at_rank_edge": [float(singular[rank - 1]), float(singular[rank])]
        if 0 < rank < len(singular)
        else None,
        "energy_in_rank": float(energy[rank - 1]) if rank else 0.0,
    }


def merged_test(plan: dict[str, Any]) -> dict[str, Any]:
    """Decide how checkpoint A must be loaded, from the weights themselves."""
    sft = plan["checkpoints"]["sft"]
    grpo = plan["checkpoints"]["grpo"]
    root = RemoteShards(sft["hf_repo"], sft["revision"])
    rl = RemoteShards(grpo["hf_repo"], grpo["revision"])
    base = RemoteShards(BASE_REPO)
    adapter_cfg = root.json("lora_adapter/adapter_config.json")
    scale = adapter_cfg["lora_alpha"] / adapter_cfg["r"]
    norm = np.linalg.norm

    report: dict[str, Any] = {
        "adapter_base_model": adapter_cfg.get("base_model_name_or_path"),
        "lora_scale": scale,
        "lora_rank": adapter_cfg["r"],
        "tensors": {},
    }
    for key in TARGETED:
        w_root, w_base, w_rl = root.tensor(key), base.tensor(key), rl.tensor(key)
        prefix = f"base_model.model.{key[: -len('.weight')]}"
        lora_a = root.tensor(
            f"{prefix}.lora_A.weight", "lora_adapter/adapter_model.safetensors"
        )
        lora_b = root.tensor(
            f"{prefix}.lora_B.weight", "lora_adapter/adapter_model.safetensors"
        )
        delta_adapter = scale * (lora_b @ lora_a)
        delta_root = w_root - w_base
        report["tensors"][key] = {
            "rel_root_minus_base": float(norm(delta_root) / norm(w_base)),
            "cos_root_delta_vs_adapter": float(
                (delta_root * delta_adapter).sum()
                / (norm(delta_root) * norm(delta_adapter))
            ),
            "root_minus_base": effective_rank(delta_root, w_root),
            "rel_grpo_minus_root": float(norm(w_rl - w_root) / norm(w_root)),
            "rel_grpo_minus_root_plus_adapter": float(
                norm(w_rl - (w_root + delta_adapter)) / norm(w_root)
            ),
        }
    untargeted = root.tensor(UNTARGETED)
    report["untargeted_root_equals_base"] = bool(
        np.array_equal(untargeted, base.tensor(UNTARGETED))
    )

    llm = report["tensors"][TARGETED[0]]
    checks = {
        # A merged LoRA leaves every non-targeted weight exactly at the base.
        "untargeted_weight_bit_identical_to_base": report[
            "untargeted_root_equals_base"
        ],
        # ...and moves the targeted ones by a low-rank delta.
        "targeted_weight_differs_from_base": llm["rel_root_minus_base"] > 1e-3,
        "delta_is_low_rank": all(
            0 < t["root_minus_base"]["rank"] <= 4 * adapter_cfg["r"]
            for t in report["tensors"].values()
        ),
        # B was trained from the root as published: it sits next to the root,
        # far from root + adapter.  So applying the adapter would not even
        # give B's starting point.
        "grpo_started_from_root_not_root_plus_adapter": all(
            t["rel_grpo_minus_root"] < 0.1 * t["rel_grpo_minus_root_plus_adapter"]
            for t in report["tensors"].values()
        ),
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    report["verdict"] = (
        "root weights are merged: load the root with is_lora=false and do not "
        "apply lora_adapter/"
        if report["passed"]
        else "UNRESOLVED: do not start S0 until the loading convention is settled"
    )
    return report


# -- local verification ------------------------------------------------------


def _sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify_local(
    path: str | Path, ckpt: dict[str, Any], unnorm_key: str, hash_shards: bool = True
) -> dict[str, Any]:
    """Check a downloaded checkpoint against the pinned revision."""
    root = Path(path)
    problems: list[str] = []
    stats_path = root / "dataset_statistics.json"
    keys: list[str] = []
    if stats_path.is_file():
        keys = list(json.loads(stats_path.read_text(encoding="utf-8")))
        if keys != [unnorm_key]:
            problems.append(f"dataset_statistics keys {keys} != [{unnorm_key!r}]")
    else:
        problems.append("dataset_statistics.json missing")
    if not (root / "config.json").is_file():
        problems.append("config.json missing")

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        referenced = set(
            json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values()
        )
        if referenced != set(ckpt["shards"]):
            problems.append(
                f"index references {sorted(referenced)}, plan pins {sorted(ckpt['shards'])}"
            )
    else:
        problems.append("model.safetensors.index.json missing")

    shards: dict[str, str] = {}
    for name, expected in ckpt["shards"].items():
        shard = root / name
        if not shard.is_file():
            problems.append(f"{name} missing")
            continue
        if hash_shards:
            actual = _sha256(shard)
            shards[name] = "ok" if actual == expected else f"MISMATCH {actual}"
            if actual != expected:
                problems.append(f"{name} sha256 {actual} != pinned {expected}")
    if (root / "lora_adapter").exists():
        # Harmless on disk, but nothing may point lora_path at it.
        problems.append("lora_adapter/ present; the plan excludes it from download")
    return {
        "path": str(root),
        "unnorm_keys": keys,
        "shards": shards,
        "problems": problems,
        "passed": not problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("merged-test", help="remote merged-vs-adapter test for checkpoint A")
    verify = sub.add_parser("verify", help="verify a local checkpoint copy")
    verify.add_argument("--ckpt", required=True, choices=["sft", "grpo"])
    verify.add_argument("--path", required=True)
    verify.add_argument("--no-hash", action="store_true", help="skip shard sha256")
    parser.add_argument("--plan", default=str(PLAN_PATH))
    args = parser.parse_args(argv)

    plan = load_plan(Path(args.plan))
    if args.command == "merged-test":
        report = merged_test(plan)
    else:
        report = verify_local(
            args.path,
            plan["checkpoints"][args.ckpt],
            plan["unnorm_key"],
            hash_shards=not args.no_hash,
        )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
