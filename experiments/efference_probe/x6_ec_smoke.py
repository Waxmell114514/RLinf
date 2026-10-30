"""X6-EC smoke: the efference token on the real ``rlinf`` OFT class (GPU).

Pre-registered in ``PREREG_X6.md`` (section "Wrapper smoke").  On one batch of
real observations from the checkpoint the harness loads:

(a) the plain class: greedy ``predict_action_batch`` logprobs against a
    ``default_forward`` replay of the same tokens (the reference agreement);
(b) the adopted class with a random projector output layer: the same check.
    The two slices are counted from opposite ends of the sequence, so a
    misplaced token or logit row shows up as an order-1 mismatch;
(c) the adopted class at its zero init: greedy token agreement with (a),
    reported only (a zero token still takes attention mass);
(d) whether the checkpoint's norm stats carry the 8-d proprio bounds that
    :class:`ec_policy.EfferenceHistory` normalises the state with.

Pass iff (a) and (b) agree within the capture check's tolerance
(0.05 + 0.02 x max |logprob|) and every action is finite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efference_probe.config import load_config  # noqa: E402

GREEDY = {"do_sample": False, "temperature": 1.0, "top_k": -1, "top_p": 1.0}
ATOL, RTOL = 0.05, 0.02
IN_DIM = 136


def _agreement(model, obs: dict[str, Any], efference=None) -> dict[str, Any]:
    import torch

    model_obs = dict(obs)
    if efference is not None:
        model_obs["efference"] = efference
    with torch.no_grad():
        actions, result = model.predict_action_batch(env_obs=model_obs, **GREEDY)
        replay = model.default_forward(
            forward_inputs=dict(result["forward_inputs"]),
            compute_logprobs=True,
            temperature=1.0,
            top_k=-1,
        )
    ours = replay["logprobs"].float().reshape(result["prev_logprobs"].shape)
    theirs = result["prev_logprobs"].float()
    diff = float((ours - theirs).abs().max().item())
    tolerance = ATOL + RTOL * float(theirs.abs().max().item())
    actions = np.asarray(
        actions.detach().cpu() if hasattr(actions, "detach") else actions
    )
    return {
        "max_abs_logprob_diff": diff,
        "tolerance": tolerance,
        "agrees": diff <= tolerance,
        "actions_finite": bool(np.isfinite(actions).all()),
        "action_shape": list(actions.shape),
        "tokens": result["forward_inputs"]["action_tokens"].detach().cpu().numpy(),
    }


def run(cfg) -> dict[str, Any]:
    import torch

    from efference_probe import ec_policy
    from efference_probe.harness import EfferenceHarness

    harness = EfferenceHarness(cfg)
    harness.setup()
    model = harness.model
    task = int(cfg.env.task_ids[0])
    ids = harness.reset_state_ids_for_task(task)[: cfg.env.num_envs]
    obs, _ = harness.env.reset(env_idx=np.arange(len(ids)), reset_state_ids=ids)
    n = len(ids)
    out: dict[str, Any] = {"checkpoint": cfg.model.model_path, "task": task, "n": n}

    plain = _agreement(model, obs)
    rng = np.random.default_rng(0)
    efference = rng.uniform(-1, 1, size=(n, IN_DIM)).astype(np.float32)
    ec_policy.adopt(model, IN_DIM)
    out["token_row"] = 1 + model._num_patch_tokens()
    zero = _agreement(model, obs, efference)
    with torch.no_grad():
        fc2 = model.efference_projector.fc2
        fc2.weight.normal_(
            0.0, 0.02, generator=torch.Generator(fc2.weight.device).manual_seed(0)
        )
    trained = _agreement(model, obs, efference)

    out["a_plain"] = {k: v for k, v in plain.items() if k != "tokens"}
    out["b_adopted_random_fc2"] = {k: v for k, v in trained.items() if k != "tokens"}
    out["c_zero_init_token_agreement_with_plain"] = float(
        np.mean(zero["tokens"] == plain["tokens"])
    )
    stats = model.norm_stats.get(cfg.model.unnorm_key, {}).get("proprio")
    out["d_proprio_stats"] = (
        {k: list(map(float, stats[k])) for k in ("q01", "q99") if k in stats}
        if stats
        else None
    )
    out["passed"] = bool(
        plain["agrees"]
        and trained["agrees"]
        and plain["actions_finite"]
        and trained["actions_finite"]
        and zero["actions_finite"]
    )
    harness.teardown()
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    try:
        result = run(cfg)
    except Exception as error:  # noqa: BLE001 - record, then fail
        result = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
