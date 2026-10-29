"""X1: observables the 8-dim end-effector state omits (pure numpy).

``LiberoEnv`` hands the policy ``states`` = [eef pos (3), eef axis-angle (3),
gripper qpos (2)] and keeps the raw robosuite observation of every env in
``env.current_raw_obs``.  The raw dict also carries the arm's joint positions
and LIBERO's object observables, which the camera sees and the 8-dim state
does not.  These helpers read them without touching ``rlinf/``
(PREREG_X1_X4.md, X1 "New observables").
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

JOINT_KEY = "robot0_joint_pos"
EEF_KEY = "robot0_eef_pos"
EEF_TOLERANCE = 1e-6


def object_keys(raw_obs: dict[str, Any]) -> list[str]:
    """World-frame object pose keys of one raw observation, in a fixed order.

    LIBERO's object observables are ``<object>_pos`` and ``<object>_quat``.
    Robot keys (``robot0_*``) are the policy's own state, and the
    ``<object>_to_robot0_eef_*`` keys are relative poses: a function of the
    object pose and the eef pose, both already logged.
    """
    keys = [
        key
        for key in raw_obs
        if key.endswith(("_pos", "_quat"))
        and not key.startswith("robot0_")
        and "_to_robot0" not in key
    ]
    return sorted(keys)


def joint_positions(raw_obs: Sequence[dict[str, Any]]) -> np.ndarray:
    """``[n_envs, 7]`` arm joint positions."""
    return np.stack([np.asarray(obs[JOINT_KEY], dtype=np.float64) for obs in raw_obs])


def object_poses(raw_obs: Sequence[dict[str, Any]], keys: list[str]) -> np.ndarray:
    """``[n_envs, sum of key widths]``: the keys' values concatenated in order."""
    rows = []
    for obs in raw_obs:
        missing = [key for key in keys if key not in obs]
        if missing:
            raise KeyError(f"raw observation lacks object keys {missing}")
        rows.append(
            np.concatenate(
                [np.asarray(obs[key], dtype=np.float64).ravel() for key in keys]
            )
        )
    return np.stack(rows)


def eef_mismatch(raw_obs: Sequence[dict[str, Any]], states: np.ndarray) -> float:
    """Max |raw eef position - states[:, :3]| over the given envs.

    The logged 8-dim state is built from the same raw observation, so anything
    above float32 rounding means the extra observables were read from a
    different moment than the state they are logged next to.
    """
    raw = np.stack([np.asarray(obs[EEF_KEY], dtype=np.float64) for obs in raw_obs])
    return float(np.max(np.abs(raw - np.asarray(states, dtype=np.float64)[:, :3])))


def check_eef(
    raw_obs: Sequence[dict[str, Any]], states: np.ndarray, where: str
) -> None:
    mismatch = eef_mismatch(raw_obs, states)
    if mismatch > EEF_TOLERANCE:
        raise RuntimeError(
            f"extra observables out of step with states ({where}): "
            f"max |eef pos difference| = {mismatch:.3g} > {EEF_TOLERANCE}"
        )
