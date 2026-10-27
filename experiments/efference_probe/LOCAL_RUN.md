# Running E4's CPU-only probe rounds locally

What is left of E4 needs no GPU: the ten goal-suite MLP rounds and a re-run of
the libero_long GRPO linear round (Amendment 2 in `logbook.md`). On Modal the
MLP rounds were preempted and restarted from scratch, so they run on your own
machine. `e4_local.py` fetches what they read from the Modal Volume, runs each
round through the unchanged driver (`e4_wave.py probe`: same gates, completion
markers, watchdog and logbook lines), and puts the results back. Every Modal
step is a client-side Volume read or write; no Modal container runs, so it
costs nothing.

## Machine

- **OS:** Windows 10/11 x64, Linux or macOS.
- **CPU and memory:** at least 16 logical CPUs and 32 GB RAM recommended (8 / 24 GB minimum). The runner picks `--jobs` as min(CPUs, 24, (RAM − 6 GB) / 1.2 GB).
- **Disk:** about 20 GB free.
- **Time:** about 2 h per MLP round with 24 workers, about 2.8 h with 16, so 20–30 h in all.
- **Keep it awake:** the watchdog counts wall-clock time, so a machine that sleeps for hours wakes up past its limit and the round is stopped. On Windows, set Settings → System → Power → Sleep to *Never* for the run.

## Setup (PowerShell)

```powershell
winget install --id=astral-sh.uv -e        # or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone -b claude/lucid-archimedes-6f31ty https://github.com/Waxmell114514/RLinf C:\rlinf
cd C:\rlinf
uv venv -p 3.11 .venv-e4
.venv-e4\Scripts\activate
uv pip install -r experiments\efference_probe\requirements-local.txt
python -m modal token set --token-id <id> --token-secret <secret>   # the workspace that holds efference-e4
```

A short checkout path such as `C:\rlinf` keeps file paths under Windows' 260-character limit.

## Run

```powershell
python experiments\efference_probe\e4_local.py check      # machine, packages, Volume, the 11 rounds
python experiments\efference_probe\e4_local.py download   # ~14 GB, resumable
python experiments\efference_probe\e4_local.py run        # 20-30 h, resumable
python experiments\efference_probe\e4_local.py upload     # results back to the Volume
```

- **What `check` should list:** 11 rounds, in this order:
  1. `mlp` on `e4_grpo_goal_swap_s1` and `e4_grpo_goal_swap_s2`;
  2. `mlp` on the eight `e4_{sft,grpo}_goal_{mirror,freeze}_s{1,2}` runs;
  3. `linear` on `e4_grpo_long_swap_s0`.

  Anything else means the data root is not what the plan expects; stop there.
- **Interrupting and resuming:**
  - `Ctrl+C` stops at any time.
  - Running `run` again skips the rounds that finished.
  - An interrupted round starts over; its half-written directory is kept as `*.partial_<UTC>`.
- **Watchdog:**
  - With fewer than 24 workers, every expected time, and so the watchdog, is scaled by 24 / jobs, as the lock's speed factor does for collections.
  - Each round's status file records the factor, and the logbook part records the machine and package versions.
- **Do not change anything on the Volume while this runs.** `upload` overwrites files whose size differs.
- **Afterwards:** the results are merged and reported from the Volume, with `modal_e4.py::pull` and then `e4_report.py`.
