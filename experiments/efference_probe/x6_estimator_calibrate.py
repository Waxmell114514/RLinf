"""X6 R1 step 2, offline: calibrate and select the analytic fault estimator.

Calibration and selection use the S3 training collection (reset states 30-49).
The selected setting is then replayed, unchanged, on S1 (the evaluation states
0-29, uncompensated faults) as a held-out check.  Open-loop replay: the logged
sent actions and displacements drive the estimator; nothing is re-simulated.

Selection rule (fixed before looking at S1 with it): over the grid, the
setting with the lowest mean detection call on S3's faulted episodes of the
evaluation's kind (inversions, and biases of magnitude >= 0.25; the evaluation
cell is bias_x 0.3), among those whose clean-episode false-alarm rate on S3 is
<= ``--max-false-alarm``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efference_probe import faults  # noqa: E402
from efference_probe.fault_estimator import (  # noqa: E402
    EstimatorParams,
    calibrate,
    replay,
)

CHUNK, DIM = 8, 7


def load(run_dir: Path, sent_column: str) -> pd.DataFrame:
    cols = [
        "episode_id",
        "call_idx",
        "fault",
        sent_column,
        "states_before",
        "states_after",
    ]
    calls = pd.read_parquet(run_dir / "calls.parquet", columns=cols)
    return calls.sort_values(["episode_id", "call_idx"]).reset_index(drop=True)


def arrays(calls: pd.DataFrame, sent_column: str):
    sent = np.stack(calls[sent_column].to_numpy()).reshape(len(calls), CHUNK, DIM)
    before = np.stack(calls["states_before"].to_numpy())
    after = np.stack(calls["states_after"].to_numpy())
    ds = after - before
    same = calls["episode_id"].to_numpy()[1:] == calls["episode_id"].to_numpy()[:-1]
    prev_chunk = np.zeros_like(sent)
    prev_chunk[1:][same] = sent[:-1][same]
    ds_prev = np.zeros_like(ds)
    ds_prev[1:][same] = ds[:-1][same]
    return sent, prev_chunk, ds_prev, ds


def truth(spec: str) -> tuple[np.ndarray, np.ndarray]:
    """(sign [2], bias [2]) of a fault spec."""
    fault = faults.parse(spec)
    sign, bias = np.ones(2), np.zeros(2)
    if fault.family == "invert_x":
        sign[0] = -1
    elif fault.family == "invert_y":
        sign[1] = -1
    elif fault.family in ("bias_x", "bias"):
        bias = np.asarray(fault.bias_vector, dtype=float)
    return sign, bias


def score(calls: pd.DataFrame, signs: np.ndarray, biases: np.ndarray) -> dict:
    """Per-episode false alarms (clean), detection call and errors (faulted)."""
    out: dict = {"clean": {}, "faulted": {}}
    calls = calls.assign(row=np.arange(len(calls)))
    alarms, alarm_calls, n_clean_calls = [], 0, 0
    families: dict[str, list] = {}
    for episode, group in calls.groupby("episode_id", sort=False):
        rows = group["row"].to_numpy()
        spec = group["fault"].iloc[0]
        active = (signs[rows] < 0).any(1) | (biases[rows] != 0).any(1)
        if spec == "clean":
            alarms.append(bool(active.any()))
            alarm_calls += int(active.sum())
            n_clean_calls += len(rows)
            continue
        sign_t, bias_t = truth(spec)
        kind = faults.parse(spec).family
        family = kind
        if kind == "bias" and np.linalg.norm(bias_t) >= 0.25:
            family = "bias_large"
        ok_sign = (signs[rows] == sign_t).all(1)
        if kind in ("bias", "bias_x"):
            err = np.linalg.norm(biases[rows] - bias_t, axis=1)
            good = ok_sign & (err <= 0.5 * np.linalg.norm(bias_t))
        else:
            err = np.zeros(len(rows))
            good = ok_sign & (biases[rows] == 0).all(1)
        first = int(np.argmax(good)) if good.any() else len(rows)
        wrong_bias = (
            float(np.abs(biases[rows]).max()) if kind.startswith("invert") else 0.0
        )
        families.setdefault(family, []).append(
            {
                "detect_call": first,
                "detected": bool(good.any()),
                "final_err": float(err[-1]),
                "wrong_sign_calls": int((~ok_sign).sum()),
                "max_wrong_bias": wrong_bias,
                "calls": len(rows),
            }
        )
    out["clean"] = {
        "episodes": len(alarms),
        "episode_false_alarm": float(np.mean(alarms)) if alarms else None,
        "call_false_alarm": alarm_calls / max(n_clean_calls, 1),
    }
    for family, items in families.items():
        frame = pd.DataFrame(items)
        out["faulted"][family] = {
            "episodes": len(frame),
            "detected": float(frame["detected"].mean()),
            "detect_call_mean": float(frame["detect_call"].mean()),
            "detect_call_median": float(frame["detect_call"].median()),
            "final_bias_err_mean": float(frame["final_err"].mean()),
            "max_wrong_bias_mean": float(frame["max_wrong_bias"].mean()),
            "wrong_sign_call_frac": float(
                frame["wrong_sign_calls"].sum() / frame["calls"].sum()
            ),
        }
    return out


SELECT = ("bias_large", "invert_x", "invert_y")


def mean_detect(scored: dict) -> float:
    fam = [f for name, f in scored["faulted"].items() if name in SELECT]
    return float(np.mean([f["detect_call_mean"] for f in fam]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, help="S3 collection run dir")
    parser.add_argument("--heldout", required=True, help="S1 sweep run dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-false-alarm", type=float, default=0.02)
    parser.add_argument("--kind", choices=["joint", "select"], default="joint")
    args = parser.parse_args(argv)

    train = load(Path(args.train), "a_sent")
    sent, prev_chunk, ds_prev, ds = arrays(train, "a_sent")
    clean = (train["fault"] == "clean").to_numpy()
    plant = calibrate(sent[clean], prev_chunk[clean], ds_prev[clean], ds[clean])

    if args.kind == "select":
        settings = [
            {"kind": "select", "p_act": p_act, "bias_sd": bias_sd}
            for p_act, bias_sd in itertools.product(
                [0.8, 0.9, 0.95, 0.99], [0.15, 0.25]
            )
        ]
    else:
        settings = [
            {"z_sign": z_sign, "z_bias": z_bias, "prior_sign_sd": prior_sign_sd}
            for z_sign, z_bias, prior_sign_sd in itertools.product(
                [2.0, 3.0, 4.0], [2.0, 2.5, 3.0, 3.5, 4.0], [0.15, 0.3]
            )
        ]
    candidates = []
    for setting in settings:
        params = EstimatorParams(**plant, **setting)
        signs, biases = replay(
            params,
            sent,
            ds,
            train["episode_id"].to_numpy(),
            train["call_idx"].to_numpy(),
        )
        scored = score(train, signs, biases)
        candidates.append({"params": params.to_dict(), "train": scored})
        print(
            f"{setting}: "
            f"clean FA {scored['clean']['episode_false_alarm']:.3f} "
            f"mean detect {mean_detect(scored):.2f}",
            flush=True,
        )
    ok = [
        cand
        for cand in candidates
        if cand["train"]["clean"]["episode_false_alarm"] <= args.max_false_alarm
    ]
    if not ok:
        print("no setting meets the false-alarm bound", file=sys.stderr)
        chosen = None
    else:
        chosen = min(ok, key=lambda cand: mean_detect(cand["train"]))
        params = EstimatorParams.from_dict(chosen["params"])
        held = load(Path(args.heldout), "a_cmd_env")
        h_sent, _, _, h_ds = arrays(held, "a_cmd_env")
        signs, biases = replay(
            params,
            h_sent,
            h_ds,
            held["episode_id"].to_numpy(),
            held["call_idx"].to_numpy(),
        )
        chosen["heldout"] = score(held, signs, biases)
    result = {
        "calibration": plant,
        "max_false_alarm": args.max_false_alarm,
        "chosen": chosen,
        "candidates": candidates,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    if chosen:
        print(
            json.dumps({k: chosen[k] for k in ("params", "train", "heldout")}, indent=2)
        )
    return 0 if chosen else 1


if __name__ == "__main__":
    sys.exit(main())
