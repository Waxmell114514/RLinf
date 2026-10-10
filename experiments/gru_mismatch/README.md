# E1 — hijack mismatch in the GRU of "From Prediction to Self"

Line A, experiment 1 of the wave-3 plan (`../efference_probe/PREREG_wave3.md`).
Question: in Ye (2026)'s minimal agent, does *keeping the command* make a
hijacked action readable, or is a comparator also needed?

CPU only, about 12 min per (interface, seed) job at full scale.

## What runs

1. **Training — a faithful port of exp3/exp4** (`port.py`). Constants, module
   construction order and RNG consumption follow the source, so
   `interface="none"` is exp3 and `"trace_abs"` is exp4. The port reproduces
   exp4's trailing recall at the source seed exactly (56.5%). Two further
   interfaces: `trace_signed` (the trace keeps the command's sign) and
   `command` (the instantaneous signed command as GRU input).
2. **Phase 3 — frozen, with hijacks** (`mismatch.py`). 40k steps; on a
   hijacked step the efference input gets the command `a` while the world
   moves by `γ·a'`: swap (`a'` an independent draw), mirror (`a' = −a`) or
   freeze (`a' = 0`). The three arms replay the same world.
3. **Probes** (`analyze.py`) — the VLA pipeline's own sample builder and
   `run_probe`: phase-matched negatives, 5-fold CV grouped by burst, nested C,
   selection-aware shuffled-label floor, permutation null.
4. **Aggregate** (`aggregate.py`) — cross-seed table and the pre-registered
   verdict for the primary cell (`trace_abs` × swap).

Two facts about the source protocol to keep in mind when reading results:
phase 1 never trains the GRU (the loss reaches only the readout), and phase 2
trains it only through the supervised active/trailing/quiet head.

## Commands

```bash
# reproduction gate (exp4 at the source seed; train only)
python experiments/gru_mismatch/run_gru.py --interfaces trace_abs none \
    --seeds 42 --transforms --jobs 2 --out experiments/gru_mismatch/results/repro

# the pre-registered run: 4 interfaces x 5 seeds x 3 arms
python experiments/gru_mismatch/run_gru.py --jobs 4 \
    --out experiments/gru_mismatch/results/e1

# smoke test (the source's --quick: step counts / 5)
python experiments/gru_mismatch/run_gru.py --quick --interfaces trace_abs \
    --seeds 42 --permutations 5 --out /tmp/gru_quick

pytest experiments/gru_mismatch/tests -q
```

Each job writes `<out>/<interface>/seed<N>/{run.json,probe_results.csv,log.txt}`;
`<out>/summary.md` and `summary.csv` hold the aggregate.

## Attribution

Training protocol ported from
[yefeifei-tech/prediction-to-self-journal](https://github.com/yefeifei-tech/prediction-to-self-journal)
under the MIT license; see `LICENSE-prediction-to-self`.
