# X4 report — external comparator in the loop

Pre-registered in `PREREG_X1_X4.md` (X4). Every number is read from the runs' manifests and `calls.parquet`.

| ckpt | G | B_mon [CI] | B_orc [CI] | rule M | rule O | reading |
|---|---|---|---|---|---|---|
| sft | 0.167 | +0.015 [-0.083, +0.117] | +0.002 [-0.093, +0.098] | False | False | undo does not recover the cost; X4 is uninformative about the comparator |
| grpo | 0.219 | +0.059 [-0.013, +0.132] | +0.083 [+0.017, +0.150] | False | False | undo does not recover the cost; X4 is uninformative about the comparator |

```json
[
  {
    "ckpt": "sft",
    "G": 0.167,
    "B_mon": {
      "point": 0.01523897483260217,
      "ci_low": -0.08250596475025018,
      "ci_high": 0.1168494574001384,
      "n_treated": 183,
      "n_control": 213
    },
    "B_orc": {
      "point": 0.0017790956263898927,
      "ci_low": -0.09326105757351125,
      "ci_high": 0.0979633061527057,
      "n_treated": 190,
      "n_control": 213
    },
    "rule_M": false,
    "rule_O": false,
    "reading": "undo does not recover the cost; X4 is uninformative about the comparator",
    "fp_cost": {
      "point": -0.01388888888888884,
      "ci_low": -0.17804232804232803,
      "ci_high": 0.13888888888888884,
      "n_treated": 56,
      "n_control": 108
    },
    "online_fpr": 0.023574947220267418,
    "online_tpr": 0.7317339149400218,
    "n_off_self_calls": 5684,
    "n_off_hijack_calls": 917,
    "undo_error_median_cm": 1.3866926077753305,
    "n_undos": 1330,
    "off_probe_success": 0.49295774647887325,
    "e4_probe_success": 0.477,
    "sanity_flag": false,
    "hijacks_per_probe_episode": {
      "off": 4.305164319248826,
      "monitor": 4.2896174863387975,
      "oracle": 4.126315789473685
    },
    "n_episodes": 750,
    "n_censored": 0
  },
  {
    "ckpt": "grpo",
    "G": 0.219,
    "B_mon": {
      "point": 0.059112304715700725,
      "ci_low": -0.01269167460798759,
      "ci_high": 0.13169597736001598,
      "n_treated": 238,
      "n_control": 291
    },
    "B_orc": {
      "point": 0.08334368401440817,
      "ci_low": 0.016755682523910048,
      "ci_high": 0.15037469465490838,
      "n_treated": 249,
      "n_control": 291
    },
    "rule_M": false,
    "rule_O": false,
    "reading": "undo does not recover the cost; X4 is uninformative about the comparator",
    "fp_cost": {
      "point": 0.046666666666666634,
      "ci_low": 0.020000000000000018,
      "ci_high": 0.07999999999999996,
      "n_treated": 72,
      "n_control": 150
    },
    "online_fpr": 0.018748839799517355,
    "online_tpr": 0.7535441657579062,
    "n_off_self_calls": 5387,
    "n_off_hijack_calls": 917,
    "undo_error_median_cm": 1.86247443780303,
    "n_undos": 1178,
    "off_probe_success": 0.7560137457044673,
    "e4_probe_success": 0.73,
    "sanity_flag": false,
    "hijacks_per_probe_episode": {
      "off": 3.1512027491408934,
      "monitor": 3.0210084033613445,
      "oracle": 2.8152610441767068
    },
    "n_episodes": 1000,
    "n_censored": 0
  }
]
```
