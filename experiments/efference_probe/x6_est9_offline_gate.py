"""Amendment 9 offline go/no-go (PREREG_X6.md): O1 and O2 replays.

O1: replay on F under rotation (``a_cmd_env`` was sent uncorrected), detection
(A error < 0.3 with b = 0) per episode.  O2: replay on Amendment 8's
closed-loop calls (``a_sent``): translation-on rate and applied-map changes at
calls >= 6.  O3 is the ``heldout`` block of the calibration output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import x6_estimator_calibrate as cal
from fault_estimator import EstimatorParams, replay

LATE = 6


def _load(run: str, column: str, tasks=None) -> pd.DataFrame:
    calls = pd.read_parquet(
        Path(run) / "calls.parquet",
        columns=[
            "episode_id",
            "task_id",
            "call_idx",
            "fault",
            column,
            "states_before",
            "states_after",
        ],
    )
    if tasks is not None:
        calls = calls[calls["task_id"].isin(tasks)]
    calls = calls.assign(episode_id=run + ":" + calls["episode_id"].astype(str))
    return calls.sort_values(["episode_id", "call_idx"]).reset_index(drop=True)


def _replay(params, calls, column):
    sent, _, _, ds = cal.arrays(calls, column)
    return replay(
        params, sent, ds, calls["episode_id"].to_numpy(), calls["call_idx"].to_numpy()
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    params = EstimatorParams.from_dict(json.loads(Path(args.params).read_text()))
    out: dict = {}

    rot = _load("data/x6/x6_fault_rot", "a_cmd_env")
    amaps, biases = _replay(params, rot, "a_cmd_env")
    out["O1"] = {}
    for cell in ("rotate:45", "rotate:90"):
        rows = (rot["fault"] == cell).to_numpy()
        sub = rot[rows].reset_index(drop=True)
        s = cal.score_affine(sub, amaps[rows], biases[rows])["faulted"]["rotate"]
        s["passed"] = bool(s["detected"] >= 0.90)
        out["O1"][cell] = s

    closed = pd.concat(
        [
            _load("data/x6/x6_est8_eval", "a_sent", tasks=[0, 1, 2, 3, 4]),
            _load("data/x6/x6_est8_eval_b", "a_sent"),
        ],
        ignore_index=True,
    )
    amaps, biases = _replay(params, closed, "a_sent")
    late = closed["call_idx"].to_numpy() >= LATE
    on = (biases != 0).any(1)
    same_ep = np.r_[
        False,
        closed["episode_id"].to_numpy()[1:] == closed["episode_id"].to_numpy()[:-1],
    ]
    changed = np.r_[False, (np.abs(amaps[1:] - amaps[:-1]).max((1, 2)) > 0.5)] & same_ep
    changed |= np.r_[False, on[1:] != on[:-1]] & same_ep
    out["O2"] = {}
    for cell in ("clean", "bias_x:0.3", "invert_x", "rotate:45", "rotate:90"):
        m = (closed["fault"].to_numpy() == cell) & late
        entry = {
            "late_calls": int(m.sum()),
            "translate_on": float(on[m].mean()),
            "class_change": float(changed[m].mean()),
        }
        if cell == "bias_x:0.3":
            entry["passed"] = bool(
                entry["translate_on"] >= 0.90 and entry["class_change"] <= 0.02
            )
        elif cell == "clean":
            entry["passed"] = bool(entry["class_change"] <= 0.02)
        else:
            entry["passed"] = bool(
                entry["translate_on"] <= 0.02 and entry["class_change"] <= 0.02
            )
        out["O2"][cell] = entry

    held = json.loads(Path(args.calibration).read_text())["chosen"]["heldout"]
    out["O3"] = {
        "clean_FA": held["clean"]["episode_false_alarm"],
        "invert_x_detected": held["faulted"]["invert_x"]["detected"],
    }
    out["O3"]["passed"] = bool(
        out["O3"]["clean_FA"] <= 0.02 and out["O3"]["invert_x_detected"] >= 0.90
    )
    out["go"] = bool(
        all(v["passed"] for v in out["O1"].values())
        and all(v["passed"] for v in out["O2"].values())
        and out["O3"]["passed"]
    )
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
