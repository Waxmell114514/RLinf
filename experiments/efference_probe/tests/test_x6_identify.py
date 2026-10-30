"""X6 stage 0 step 1 on synthetic transitions with a known gain and noise."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe import x6_identify as x6  # noqa: E402


def _calls(n_episodes=80, calls=12, gain=0.01, noise=0.004, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for episode in range(n_episodes):
        state = np.zeros(8)
        for call in range(calls):
            command = rng.normal(scale=0.5, size=(8, 7))
            after = state.copy()
            after[:3] += gain * command[:, :3].sum(axis=0) + rng.normal(
                scale=noise, size=3
            )
            rows.append(
                {
                    "episode_id": f"r:{episode}",
                    "call_idx": call,
                    "task_id": episode % 2,
                    "label": "SELF",
                    "a_cmd_env": command.reshape(-1),
                    "states_before": state.copy(),
                    "states_after": after,
                    "post_success": False,
                }
            )
            state = after
    return pd.DataFrame(rows)


def test_gain_and_noise_are_recovered_and_faults_scale_with_k():
    out = x6.tables(_calls())
    np.testing.assert_allclose(out["gain_cm_per_unit"], [1.0, 1.0, 1.0], atol=0.05)
    np.testing.assert_allclose(out["residual_sd_cm"], [0.4, 0.4, 0.4], atol=0.05)
    z1 = out["faults"]["K=1"]["bias_x_0.3"]["median_z"]
    z4 = out["faults"]["K=4"]["bias_x_0.3"]["median_z"]
    # 8 steps x 0.3 x 1 cm = 2.4 cm shift over 0.4 cm noise: z = 6 at K = 1.
    assert abs(z1 - 6.0) < 0.6 and abs(z4 / z1 - 2.0) < 0.2
    assert out["faults"]["K=1"]["invert_x"]["median_z"] > 3
    assert out["history_K"] == 1
    assert all(abs(c) < 0.15 for c in out["copycat_corr_xyz"])  # i.i.d. commands
