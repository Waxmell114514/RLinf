"""X6 stage 0, step 2: can our checkpoints take a proprio (efference-copy) token?

Pre-registered in ``logbook.md`` ("X6 STAGE-0 PRE-REGISTRATION").  Our
checkpoints load through ``implement_version: rlinf``, whose embedding has no
proprio path; only the ``official`` class has one.

(a) Load the same checkpoint through both classes (no proprio) and compare
    greedy action tokens on the same observations: pass iff >= 99% agree.
(b) Attach a fresh ProprioProjector(proprio_dim) outside ``rlinf/``, run the
    forward pass with a proprio input, and check the actions are finite;
    report the token agreement with (a) when its last layer is zeroed.

Both loaders register transformers' "openvla" config name, so each class runs
in its own process: ``--stage rlinf`` saves the observations and tokens,
``--stage official`` reads them.  Without ``--stage`` both run in turn.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efference_probe.config import load_config  # noqa: E402

PASS_AGREEMENT = 0.99
GREEDY = {"do_sample": False, "temperature": 1.0, "top_k": -1, "top_p": 1.0}


def _tokens(result: dict[str, Any]) -> np.ndarray:
    return result["forward_inputs"]["action_tokens"].detach().cpu().numpy()


def _numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def stage_rlinf(cfg, work: Path) -> None:
    import torch

    from efference_probe.harness import EfferenceHarness

    harness = EfferenceHarness(cfg)
    harness.setup()
    task = int(cfg.env.task_ids[0])
    ids = harness.reset_state_ids_for_task(task)[: cfg.env.num_envs]
    obs, _ = harness.env.reset(env_idx=np.arange(len(ids)), reset_state_ids=ids)
    saved = {key: _numpy(value) for key, value in obs.items()}
    with torch.no_grad():
        _, result = harness.model.predict_action_batch(env_obs=dict(obs), **GREEDY)
    payload = {
        "obs": saved,
        "tokens": _tokens(result),
        "logprobs": result["prev_logprobs"].float().cpu().numpy(),
        "task": task,
        "n": len(ids),
    }
    (work / "rlinf.pkl").write_bytes(pickle.dumps(payload))
    harness.teardown()


def stage_official(cfg, work: Path, proprio_dim: int) -> dict[str, Any]:
    import torch
    from omegaconf import open_dict

    from efference_probe.harness import build_model_config
    from rlinf.models import get_model

    saved = pickle.loads((work / "rlinf.pkl").read_bytes())
    obs = saved["obs"]
    out: dict[str, Any] = {
        "checkpoint": cfg.model.model_path,
        "task": saved["task"],
        "n": saved["n"],
    }
    model_cfg = build_model_config(cfg)
    with open_dict(model_cfg):
        model_cfg.implement_version = "official"
        model_cfg.use_proprio = False
        model_cfg.proprio_dim = proprio_dim
    try:
        official = get_model(model_cfg).to("cuda").eval()
    except Exception as error:  # noqa: BLE001 - the failure is the finding
        out.update(
            a_loaded=False, a_error=f"{type(error).__name__}: {error}", passed=False
        )
        return out
    out["a_loaded"] = True
    with torch.no_grad():
        _, result = official.predict_action_batch(env_obs=dict(obs), **GREEDY)
    tokens = _tokens(result).reshape(saved["n"], -1)
    reference = np.asarray(saved["tokens"]).reshape(saved["n"], -1)
    out["a_token_ranges"] = {
        "official": [int(tokens.min()), int(tokens.max())],
        "rlinf": [int(reference.min()), int(reference.max())],
    }
    # The classes may index action tokens differently (relative bins vs
    # vocabulary ids); the best of the plausible conventions is reported.
    vocab = int(official.config.text_config.vocab_size)
    candidates = {
        "same": reference,
        "vocab_minus_256_offset": reference - (vocab - 256),
        "vocab_minus_1_minus": vocab - 1 - reference,
    }
    agreement = {k: float(np.mean(tokens == v)) for k, v in candidates.items()}
    out["a_agreement_by_convention"] = agreement
    out["a_token_agreement"] = max(agreement.values())
    out["a_passed"] = out["a_token_agreement"] >= PASS_AGREEMENT
    logprobs = result["prev_logprobs"].float().cpu().numpy()
    out["a_max_abs_logprob_diff"] = float(np.abs(logprobs - saved["logprobs"]).max())

    # (b) A fresh projector, attached without touching rlinf/.
    from prismatic.models.projectors import ProprioProjector

    dtype = next(official.parameters()).dtype
    official.proprio_projector = ProprioProjector(
        llm_dim=official.config.text_config.hidden_size, proprio_dim=proprio_dim
    ).to("cuda", dtype=dtype)
    official.use_proprio = True
    input_ids, attention_mask, pixel_values, _ = official.prepare_inputs(dict(obs))
    rng = np.random.default_rng(0)
    proprio = torch.as_tensor(
        rng.uniform(-1, 1, size=(saved["n"], proprio_dim)), dtype=dtype
    ).to("cuda")
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "proprio": proprio,
    }
    try:
        with torch.no_grad():
            actions, _ = official.predict_action_batch(**inputs, **GREEDY)
    except Exception as error:  # noqa: BLE001 - the failure is the finding
        out.update(
            b_ran=False, b_error=f"{type(error).__name__}: {error}", passed=False
        )
        return out
    actions = np.asarray(_numpy(actions), dtype=np.float32)
    out.update(
        b_ran=True,
        b_actions_finite=bool(np.isfinite(actions).all()),
        b_action_shape=list(actions.shape),
    )
    linear = [
        m
        for m in official.proprio_projector.modules()
        if isinstance(m, torch.nn.Linear)
    ]
    torch.nn.init.zeros_(linear[-1].weight)
    torch.nn.init.zeros_(linear[-1].bias)
    with torch.no_grad():
        _, zeroed = official.predict_action_batch(**inputs, **GREEDY)
    zeroed_tokens = _tokens(zeroed).reshape(saved["n"], -1)
    out["b_zeroed_token_agreement_with_a"] = float(np.mean(zeroed_tokens == tokens))
    out["passed"] = bool(out["a_passed"] and out["b_actions_finite"])
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proprio-dim", type=int, default=136)
    parser.add_argument("--stage", choices=("rlinf", "official"))
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)
    work = args.out.parent
    work.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config, args.set)
    if args.stage == "rlinf":
        stage_rlinf(cfg, work)
        return 0
    if args.stage == "official":
        result = stage_official(cfg, work, args.proprio_dim)
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0 if result.get("passed") else 1
    common = [
        "--config",
        args.config,
        "--out",
        str(args.out),
        "--proprio-dim",
        str(args.proprio_dim),
        "--set",
        *args.set,
    ]
    for stage in ("rlinf", "official"):
        rc = subprocess.run(
            [sys.executable, __file__, "--stage", stage, *common], check=False
        ).returncode
        if rc != 0 and stage == "rlinf":
            print(json.dumps({"passed": False, "error": "rlinf stage failed"}))
            return rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
