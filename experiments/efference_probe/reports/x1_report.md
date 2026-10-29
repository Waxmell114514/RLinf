# X1 report

```json
{
  "runs": {
    "x1_sft_swap": {
      "run_id": "x1_sft_swap",
      "R1": {
        "status": "at",
        "above": [],
        "beyond_observables": [],
        "explained_by": [],
        "margins": {
          "P1": 0.01087995770552852,
          "P3": 0.004671609806098376,
          "P4": 0.01790414574967636
        },
        "floor": 0.5270722155561118,
        "p2r": 0.6833201544838607,
        "c_dstates": 0.4807529146066043,
        "c_cmd": 0.4860803604623417
      },
      "increment": -0.006208347899430144,
      "d_int": 0.007024188044147839,
      "R2": {
        "twins": {
          "C_dstates": 0.4807529146066043,
          "C_state+d": 0.5345151293191945,
          "C_scene": 0.5068548051472425,
          "C_joint": 0.507704494126395,
          "C_obs": 0.5445167662710236
        },
        "bar": 0.5845167662710237,
        "c_cmd": 0.4860803604623417,
        "cmd_in_band": true,
        "beyond": []
      },
      "R5": {
        "run": "x1_sft_swap",
        "cell": "L4/ctx_mean",
        "P1": 0.5379521732616404,
        "floor": 0.5270722155561118,
        "P1_resid": 0.5164866019654519,
        "P1_resid_folds": [
          0.5669291338582677,
          0.5,
          0.5075757575757576,
          0.5155038759689923,
          0.49242424242424243
        ],
        "flags": {
          "state-orthogonal": false,
          "state-explained": true
        },
        "reading": "state-explained"
      }
    },
    "x1_grpo_swap": {
      "run_id": "x1_grpo_swap",
      "R1": {
        "status": "above",
        "above": [
          "P3"
        ],
        "beyond_observables": [
          "P3"
        ],
        "explained_by": [],
        "margins": {
          "P1": 0.03890450198815254,
          "P3": 0.04148881121254089,
          "P4": 0.029161394218728987
        },
        "floor": 0.5486932257382431,
        "p2r": 0.6912502495175994,
        "c_dstates": 0.5212246601015813,
        "c_cmd": 0.5088268236964091
      },
      "increment": 0.002584309224388348,
      "d_int": -0.009743107769423553,
      "R2": {
        "twins": {
          "C_dstates": 0.5212246601015813,
          "C_state+d": 0.5176055459445073,
          "C_scene": 0.5185932197751015,
          "C_joint": 0.4795523099784861,
          "C_obs": 0.547021730210482
        },
        "bar": 0.587021730210482,
        "c_cmd": 0.5088268236964091,
        "cmd_in_band": true,
        "beyond": [
          "P1",
          "P3"
        ]
      },
      "R5": {
        "run": "x1_grpo_swap",
        "cell": "L20/ctx_last",
        "P1": 0.5875977277263956,
        "floor": 0.5486932257382431,
        "P1_resid": 0.5592076152771309,
        "P1_resid_folds": [
          0.5442477876106195,
          0.5707964601769911,
          0.5707964601769911,
          0.5789473684210527,
          0.53125
        ],
        "flags": {
          "state-orthogonal": false,
          "state-explained": true
        },
        "reading": "state-explained"
      }
    },
    "x1_sft_mirror": {
      "run_id": "x1_sft_mirror",
      "R1": {
        "status": "at",
        "above": [],
        "beyond_observables": [],
        "explained_by": [],
        "margins": {
          "P1": 0.021001315376853125,
          "P3": 0.021498804056824583,
          "P4": 0.03530311390720964
        },
        "floor": 0.5406998155958344,
        "p2r": 0.869184948760874,
        "c_dstates": 0.6239563741811937,
        "c_cmd": 0.48586371365395
      },
      "increment": 0.0004974886799714584,
      "d_int": 0.014301798530356513,
      "R2": {
        "twins": {
          "C_dstates": 0.6239563741811937,
          "C_state+d": 0.6065870848822461,
          "C_scene": 0.5821479042425717,
          "C_joint": 0.5678657698443936,
          "C_obs": 0.6272855952193342
        },
        "bar": 0.6672855952193343,
        "c_cmd": 0.48586371365395,
        "cmd_in_band": true,
        "beyond": []
      }
    },
    "x1_grpo_mirror": {
      "run_id": "x1_grpo_mirror",
      "R1": {
        "status": "above",
        "above": [
          "P1",
          "P3",
          "P4"
        ],
        "beyond_observables": [],
        "explained_by": [
          "C_dstates"
        ],
        "margins": {
          "P1": 0.07991970888115751,
          "P3": 0.07420741636139028,
          "P4": 0.10019146817685465
        },
        "floor": 0.5431148480010365,
        "p2r": 0.922592709098882,
        "c_dstates": 0.6379943250174532,
        "c_cmd": 0.4879781074961198
      },
      "increment": -0.005712292519767237,
      "d_int": 0.020271759295697134,
      "R2": {
        "twins": {
          "C_dstates": 0.6379943250174532,
          "C_state+d": 0.6235452589495528,
          "C_scene": 0.6055305814447755,
          "C_joint": 0.6098517702122701,
          "C_obs": 0.6750293988890386
        },
        "bar": 0.7150293988890386,
        "c_cmd": 0.4879781074961198,
        "cmd_in_band": true,
        "beyond": []
      }
    },
    "x1_sft_freeze": {
      "run_id": "x1_sft_freeze",
      "R1": {
        "status": "at",
        "above": [],
        "beyond_observables": [],
        "explained_by": [],
        "margins": {
          "P1": 0.007550891669166004,
          "P3": 0.011344601241397911,
          "P4": 0.01314172370330613
        },
        "floor": 0.536747660605052,
        "p2r": 0.8606092493228801,
        "c_dstates": 0.5480651350647824,
        "c_cmd": 0.500026287087652
      },
      "increment": 0.003793709572231907,
      "d_int": 0.005590832034140125,
      "R2": {
        "twins": {
          "C_dstates": 0.5480651350647824,
          "C_state+d": 0.5433032396328111,
          "C_scene": 0.5482855873986185,
          "C_joint": 0.5185354989974986,
          "C_obs": 0.5715271095212904
        },
        "bar": 0.6115271095212904,
        "c_cmd": 0.500026287087652,
        "cmd_in_band": true,
        "beyond": []
      }
    },
    "x1_grpo_freeze": {
      "run_id": "x1_grpo_freeze",
      "R1": {
        "status": "above",
        "above": [
          "P1",
          "P3",
          "P4"
        ],
        "beyond_observables": [],
        "explained_by": [
          "C_dstates"
        ],
        "margins": {
          "P1": 0.07278040801402641,
          "P3": 0.07621893894873322,
          "P4": 0.12804007726714217
        },
        "floor": 0.5228438477751505,
        "p2r": 0.9314091710961986,
        "c_dstates": 0.6215516260157333,
        "c_cmd": 0.5113954837918128
      },
      "increment": 0.003438530934706807,
      "d_int": 0.05525966925311576,
      "R2": {
        "twins": {
          "C_dstates": 0.6215516260157333,
          "C_state+d": 0.5951193843349325,
          "C_scene": 0.5925315077717167,
          "C_joint": 0.6093559877253218,
          "C_obs": 0.6979989910990776
        },
        "bar": 0.7379989910990776,
        "c_cmd": 0.5113954837918128,
        "cmd_in_band": true,
        "beyond": []
      }
    }
  },
  "R3": {
    "increments": {
      "x1_sft_swap": -0.006208347899430144,
      "x1_grpo_swap": 0.002584309224388348,
      "x1_sft_mirror": 0.0004974886799714584,
      "x1_grpo_mirror": -0.005712292519767237,
      "x1_sft_freeze": 0.003793709572231907,
      "x1_grpo_freeze": 0.003438530934706807
    },
    "x1_grpo_swap_bootstrap": {
      "run": "x1_grpo_swap",
      "checkpoint": "GRPO-goal",
      "family": "linear",
      "checks": {
        "P1": {
          "cell": "L20/ctx_last",
          "max_fold_diff": 0.0,
          "same_c": true
        },
        "P3": {
          "cell": "L20/ctx_last",
          "max_fold_diff": 0.0,
          "same_c": true
        },
        "C_dstates": {
          "cell": "-",
          "max_fold_diff": 0.0,
          "same_c": true
        }
      },
      "reproduced": true,
      "n": 1130,
      "n_episodes": 203,
      "P1-C_dstates": {
        "point": 0.0663716814159292,
        "ci_low": 0.028190954773869358,
        "ci_high": 0.10639353658046091
      },
      "P3-P1": {
        "point": 0.0026548672566372167,
        "ci_low": -0.01374591298835614,
        "ci_high": 0.019539057413576276
      }
    },
    "e4_grpo_bootstraps": [
      {
        "run": "e4_grpo_goal_swap_s1",
        "reproduced": true,
        "P3-P1": {
          "point": -0.0063025210084033,
          "ci_low": -0.0382226849583537,
          "ci_high": 0.0234702598988312
        }
      },
      {
        "run": "e4_grpo_goal_swap_s2",
        "reproduced": true,
        "P3-P1": {
          "point": 0.0046620046620045,
          "ci_low": -0.0136379202988791,
          "ci_high": 0.0224612961371944
        }
      }
    ],
    "reading": "no efference increment on two pairs"
  },
  "main02_reproduction": {
    "differences": {
      "P1": -0.005047826738359684,
      "P3": -0.00025617463778981886,
      "P4": 0.00897636130578816,
      "P2r": -0.027679845516139223
    },
    "within_0.04": true
  },
  "R7": {
    "x1_sft_swap": {
      "a1_within_delta": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_state_readable": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_motion_premise_fails": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_m1_compatible": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f2_no_motion_trace": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a3_above_floor": {
        "x1": false,
        "e4": true,
        "replicates": false
      },
      "f3_outcome_ok": {
        "x1": false,
        "e4": false,
        "replicates": true
      },
      "f3_command_ok": {
        "x1": true,
        "e4": false,
        "replicates": false
      },
      "f5_reading": "indeterminate (both conditions met; P1 within 0.04 of floor)"
    },
    "x1_grpo_swap": {
      "a1_within_delta": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_state_readable": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_motion_premise_fails": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_m1_compatible": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f2_no_motion_trace": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a3_above_floor": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f3_outcome_ok": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f3_command_ok": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f5_reading": "state-explained"
    },
    "x1_sft_mirror": {
      "a1_within_delta": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_state_readable": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_motion_premise_fails": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_m1_compatible": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f2_no_motion_trace": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a3_above_floor": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f3_outcome_ok": {
        "x1": false,
        "e4": false,
        "replicates": true
      },
      "f3_command_ok": {
        "x1": true,
        "e4": false,
        "replicates": false
      },
      "f5_reading": "state-explained"
    },
    "x1_grpo_mirror": {
      "a1_within_delta": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_state_readable": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_motion_premise_fails": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_m1_compatible": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f2_no_motion_trace": {
        "x1": false,
        "e4": true,
        "replicates": false
      },
      "a3_above_floor": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f3_outcome_ok": {
        "x1": false,
        "e4": true,
        "replicates": false
      },
      "f3_command_ok": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f5_reading": "partial"
    },
    "x1_sft_freeze": {
      "a1_within_delta": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_state_readable": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_motion_premise_fails": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_m1_compatible": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f2_no_motion_trace": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a3_above_floor": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f3_outcome_ok": {
        "x1": false,
        "e4": false,
        "replicates": true
      },
      "f3_command_ok": {
        "x1": true,
        "e4": false,
        "replicates": false
      },
      "f5_reading": "indeterminate (both conditions met; P1 within 0.04 of floor)"
    },
    "x1_grpo_freeze": {
      "a1_within_delta": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_state_readable": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_motion_premise_fails": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "a2_m1_compatible": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f2_no_motion_trace": {
        "x1": false,
        "e4": true,
        "replicates": false
      },
      "a3_above_floor": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f3_outcome_ok": {
        "x1": false,
        "e4": true,
        "replicates": false
      },
      "f3_command_ok": {
        "x1": true,
        "e4": true,
        "replicates": true
      },
      "f5_reading": "partial"
    },
    "pair": {
      "f4_data_effect": {
        "x1": false,
        "e4": true,
        "replicates": false
      }
    }
  },
  "R6": {
    "swap": {
      "sft_cost": 0.18719946940805832,
      "grpo_cost": 0.18310059846110005,
      "rl_minus_sft": -0.004098870946958266,
      "ci_low": -0.19968205794239022,
      "ci_high": 0.19384210658250503
    },
    "mirror": {
      "sft_cost": 0.29059365681756577,
      "grpo_cost": 0.565406976744186,
      "rl_minus_sft": 0.2748133199266203,
      "ci_low": 0.04995050298814212,
      "ci_high": 0.4884472299797638
    },
    "freeze": {
      "sft_cost": 0.025752236378422333,
      "grpo_cost": 0.11676356589147285,
      "rl_minus_sft": 0.09101132951305052,
      "ci_low": -0.1292299341904804,
      "ci_high": 0.3021585425242862
    }
  },
  "R4": {
    "x1": {
      "point": 0.03526323683815813,
      "ci_low": -0.003914927560194822,
      "ci_high": 0.0740733522117513
    },
    "e4_s1": {
      "point": 0.04959545801791787,
      "ci_low": 0.0023493448531583644,
      "ci_high": 0.09661529042038916
    },
    "e4_s2": {
      "point": 0.06393290254049733,
      "ci_low": 0.020650212284416183,
      "ci_high": 0.11037245443117534
    },
    "reading": "no RL effect at the fixed cell"
  }
}
```
