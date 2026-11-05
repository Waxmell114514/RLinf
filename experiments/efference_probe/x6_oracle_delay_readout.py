"""Amendment 10 readout (PREREG_X6.md): the delayed-oracle curve, C1 and C2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import x6_ec_report as rep

FAULTS = ("invert_x", "rotate:90")
KS = range(6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/x6/x6_oracle_delay")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(rep.SEED)
    frame = rep.episodes(Path(args.run))
    out: dict = {"curve": {}, "rules": {}}
    for fault in FAULTS:
        base = rep.success_by_pair(frame, f"{fault}#0")
        curve = {}
        for k in KS:
            s = rep.success_by_pair(frame, f"{fault}#{k}")
            draws = s.to_numpy()[rng.integers(0, len(s), size=(rep.B, len(s)))]
            curve[k] = {
                "success": float(s.mean()),
                "ci_low": float(np.quantile(draws.mean(1), 0.025)),
                "ci_high": float(np.quantile(draws.mean(1), 0.975)),
                "n": int(len(s)),
                "drop_vs_k0": rep.paired_difference(base, s, rng) if k else None,
            }
        out["curve"][fault] = curve
        out["rules"][fault] = {
            "C1": bool(curve[1]["drop_vs_k0"]["ci_low"] > 0),
            "C2": bool(curve[5]["drop_vs_k0"]["ci_low"] > 0),
        }
    out["claim_supported"] = all(r["C1"] and r["C2"] for r in out["rules"].values())
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
