#!/usr/bin/env bash
#
# Rental-box runbook for the efference-probe re-collection (SPEC V2.2, T-RECOLLECT).
#
# One session on a rented 24 GB card: bootstrap -> checkpoint -> smoke -> collect
# -> archive.  Then the archive is verified *on the laptop* before the box is
# released, because of:
#
#   Doctrine 3 (download-before-release).  Never release a rental until the
#   archive is checksum-verified locally and the integrity gates pass locally.
#   Never pay for idle GPU during analysis.  Nothing exists only on rented disk.
#
# Usage on the box (in tmux):
#
#     bash experiments/efference_probe/rental_runbook.sh preflight
#     bash experiments/efference_probe/rental_runbook.sh bootstrap
#     bash experiments/efference_probe/rental_runbook.sh checkpoint
#     bash experiments/efference_probe/rental_runbook.sh smoke
#     bash experiments/efference_probe/rental_runbook.sh collect main
#     bash experiments/efference_probe/rental_runbook.sh collect mirror   # optional
#     bash experiments/efference_probe/rental_runbook.sh collect freeze   # optional
#     bash experiments/efference_probe/rental_runbook.sh archive
#
# Then on the laptop (no GPU needed):
#
#     bash experiments/efference_probe/rental_runbook.sh verify <archive.tar> <sha256>
#
# `all` chains preflight..archive for the swap run only.  Every phase is
# idempotent and safe to re-run after a disconnect.
#
# Nothing here trains anything: this is inference-only collection (SPEC 0.2).

set -euo pipefail

# ---------------------------------------------------------------------------
# Knobs.  Override from the environment, e.g. NUM_ENVS=10 bash ... collect main
# ---------------------------------------------------------------------------
REPO_PATH="${REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROBE_DIR="${REPO_PATH}/experiments/efference_probe"
DATA_ROOT="${DATA_ROOT:-${PROBE_DIR}/data}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${REPO_PATH}/archives}"
BRANCH="${BRANCH:-claude/rlinf-code-scripts-pi41dg}"

# The SFT checkpoint.  CKPT_REPO is pulled from HF only when CKPT_PATH is absent.
CKPT_PATH="${CKPT_PATH:-${REPO_PATH}/checkpoints/openvla-oft-sft-libero-goal}"
CKPT_REPO="${CKPT_REPO:-}"                    # e.g. RLinf/Openvla-oft-SFT-libero-goal-traj1
UNNORM_KEY="${UNNORM_KEY:-libero_goal_no_noops}"
TASK_SUITE="${TASK_SUITE:-libero_goal}"

# Measured on the lost cluster run: 19m29s elapsed, MaxRSS ~79 GB at num_envs=20.
# RAM, not VRAM, is the binding constraint on a rental box (SPEC V2.2).
NUM_ENVS="${NUM_ENVS:-auto}"
MIRROR_FREEZE_INITS="${MIRROR_FREEZE_INITS:-4}"   # 10 tasks x 4 = ~40 episodes
SMOKE_TIMEOUT_S="${SMOKE_TIMEOUT_S:-1800}"        # 30-minute cap on S0 (SPEC V2.1)

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
readonly START_EPOCH=$(date +%s)

log()  { printf '\n\033[1;36m[%s +%dm] %s\033[0m\n' "$(date +%H:%M:%S)" \
             "$(( ($(date +%s) - START_EPOCH) / 60 ))" "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[FAIL] %s\033[0m\n' "$*" >&2; exit 1; }

banner() {
    printf '\n\033[1;32m%s\033[0m\n' "============================================================"
    printf '\033[1;32m  %s\033[0m\n' "$*"
    printf '\033[1;32m%s\033[0m\n\n' "============================================================"
}

# ---------------------------------------------------------------------------
# preflight -- is this box actually able to do the job?  Costs seconds; a
# failure here is far cheaper than one 40 minutes into a paid collection.
# ---------------------------------------------------------------------------
resolve_num_envs() {
    if [ "${NUM_ENVS}" != "auto" ]; then
        echo "${NUM_ENVS}"
        return
    fi
    local ram_gb
    ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
    # 20 parallel LIBERO envs peaked at ~79 GB on the cluster.  Below ~96 GB of
    # RAM, halving the env count is what keeps the box off the OOM killer; the
    # run then takes ~40-60 min instead of ~20.
    if [ "${ram_gb}" -ge 96 ]; then echo 20; else echo 10; fi
}

cmd_preflight() {
    banner "PREFLIGHT"
    log "repo: ${REPO_PATH}  (branch $(git -C "${REPO_PATH}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))"
    log "git SHA: $(git -C "${REPO_PATH}" rev-parse --short HEAD 2>/dev/null || echo '?')"

    command -v nvidia-smi >/dev/null || die "no nvidia-smi: this box has no usable GPU"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

    local vram_mb
    vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "${vram_mb}" -lt 23000 ]; then
        die "VRAM ${vram_mb} MiB < 24 GB. OpenVLA-OFT LIBERO inference needs ~16 GB
     plus headroom; a smaller card will OOM mid-collection. Re-provision."
    fi
    log "VRAM ${vram_mb} MiB -- OK (need >=24 GB; inference footprint ~16 GB)"

    local ram_gb cpus disk_gb
    ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
    cpus=$(nproc)
    disk_gb=$(df -BG --output=avail "${REPO_PATH}" | tail -1 | tr -dc '0-9')
    log "RAM ${ram_gb} GB | vCPU ${cpus} | free disk ${disk_gb} GB"

    [ "${ram_gb}" -ge 60 ]  || die "RAM ${ram_gb} GB is too small even for halved parallelism (need >=60 GB)"
    [ "${disk_gb}" -ge 60 ] || die "free disk ${disk_gb} GB < 60 GB; the run writes ~1.1 GB but the
     checkpoint pull needs ~15 GB and pip needs room"
    [ "${cpus}" -ge 8 ] || warn "only ${cpus} vCPU: LIBERO env stepping will bottleneck"
    if [ "${ram_gb}" -lt 96 ]; then
        warn "RAM < 96 GB: num_envs auto-drops to $(resolve_num_envs); expect 40-60 min, not ~20"
    fi

    log "resolved num_envs = $(resolve_num_envs)"
    banner "PREFLIGHT PASSED"
}

# ---------------------------------------------------------------------------
# bootstrap -- system deps, repo at the right commit, python env
# ---------------------------------------------------------------------------
cmd_bootstrap() {
    banner "BOOTSTRAP"
    log "syncing repo to origin/${BRANCH}"
    git -C "${REPO_PATH}" fetch origin "${BRANCH}"
    git -C "${REPO_PATH}" checkout "${BRANCH}"
    git -C "${REPO_PATH}" pull --ff-only origin "${BRANCH}"
    log "HEAD is now $(git -C "${REPO_PATH}" rev-parse --short HEAD)"

    if python -c "import torch, libero" 2>/dev/null; then
        log "torch + libero already importable; skipping install"
    else
        log "installing openvla-oft + libero (this is the slow phase, ~20-40 min)"
        log "NOTE: install.sh wants REPO_PATH set; it is: ${REPO_PATH}"
        ( cd "${REPO_PATH}" && bash requirements/install.sh embodied \
            --model openvla-oft --env libero )
    fi

    log "installing the analysis-side extras"
    python -m pip install --quiet scikit-learn matplotlib pandas pyarrow pytest

    log "offline test suite (proves the analysis half survived the install)"
    ( cd "${REPO_PATH}" && python -m pytest experiments/efference_probe/tests/test_offline.py -q ) \
        || die "offline tests failed on this box -- fix before spending GPU time"
    banner "BOOTSTRAP DONE"
}

# ---------------------------------------------------------------------------
# checkpoint -- ~15 GB pull, counted against the session budget
# ---------------------------------------------------------------------------
cmd_checkpoint() {
    banner "CHECKPOINT"
    if [ -d "${CKPT_PATH}" ] && [ -n "$(ls -A "${CKPT_PATH}" 2>/dev/null)" ]; then
        log "checkpoint already present at ${CKPT_PATH}"
        du -sh "${CKPT_PATH}"
        return 0
    fi
    [ -n "${CKPT_REPO}" ] || die "no checkpoint at ${CKPT_PATH} and CKPT_REPO is unset.
     Set CKPT_REPO=<hf-org>/<model> (SFT, matching ${TASK_SUITE}), or upload a
     local copy to ${CKPT_PATH}."

    log "pulling ${CKPT_REPO} via ${HF_ENDPOINT} (~15 GB)"
    python -m pip install --quiet "huggingface_hub[cli]"
    mkdir -p "${CKPT_PATH}"
    huggingface-cli download "${CKPT_REPO}" --local-dir "${CKPT_PATH}" \
        --local-dir-use-symlinks False
    du -sh "${CKPT_PATH}"
    banner "CHECKPOINT READY"
}

# ---------------------------------------------------------------------------
# Shared env for anything that touches LIBERO (mirrors evaluations/run_eval.sh)
# ---------------------------------------------------------------------------
setup_sim_env() {
    export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
    export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
    # Standard LIBERO only.  Pro/Plus perturbation modes are out of scope (SPEC 1).
    export LIBERO_TYPE=standard
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
}

# ---------------------------------------------------------------------------
# smoke -- S0 gate, hard-capped at 30 minutes.  This checks the *box*, not the
# science: the pipeline itself is already proven (SPEC V2, Plan R).
# ---------------------------------------------------------------------------
cmd_smoke() {
    banner "S0 SMOKE (hard cap ${SMOKE_TIMEOUT_S}s)"
    setup_sim_env
    rm -rf "${DATA_ROOT}/smoke01"

    if ! timeout --signal=INT "${SMOKE_TIMEOUT_S}" \
        python "${PROBE_DIR}/run_collect.py" \
            --config "${PROBE_DIR}/configs/smoke.yaml" \
            --set "model.model_path=${CKPT_PATH}" \
                  "model.unnorm_key=${UNNORM_KEY}" \
                  "env.task_suite_name=${TASK_SUITE}"
    then
        die "smoke run failed or blew the ${SMOKE_TIMEOUT_S}s cap.
     Per SPEC V2 risk table: stop paying, debug offline or switch plans.
     Do not grind on a metered box."
    fi

    log "checking smoke acceptance"
    python - <<PY || die "smoke output failed its acceptance checks"
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

run = Path("${DATA_ROOT}/smoke01")
calls = pd.read_parquet(run / "calls.parquet")
manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
verification = manifest.get("verification", {})

print(f"rows={len(calls)} episodes={calls.episode_id.nunique()}")
print(f"labels={calls.label.value_counts().to_dict()}")
print(f"verification={verification}")

ok = True
if not verification.get("passed"):
    print("FAIL: indexing check did not pass"); ok = False
hidden = sorted((run / "hidden").glob("*.npz"))
if not hidden:
    print("FAIL: no hidden archives"); ok = False
else:
    with np.load(hidden[0]) as a:
        print(f"hidden shape={a['h'].shape} dtype={a['h'].dtype}")
        if a["h"].ndim != 4:
            print("FAIL: hidden is not [calls, layer, pool, dim]"); ok = False
if calls.isna().any().any():
    print(f"FAIL: null cells in {list(calls.columns[calls.isna().any()])}"); ok = False
if "HIJACK" not in set(calls.label):
    print("FAIL: no hijacks were scheduled"); ok = False
sys.exit(0 if ok else 1)
PY
    banner "S0 SMOKE PASSED -- the box is good, proceed to collection"
}

# ---------------------------------------------------------------------------
# collect -- the actual data.  `main` is the primary swap run; mirror/freeze
# are the reduced per-transform controls that swap-only data cannot provide.
# ---------------------------------------------------------------------------
cmd_collect() {
    local which="${1:-main}"
    local config extra=()
    local envs; envs=$(resolve_num_envs)

    case "${which}" in
        main)   config="${PROBE_DIR}/configs/main.yaml" ;;
        mirror) config="${PROBE_DIR}/configs/main_mirror.yaml"
                extra+=("env.init_states_per_task=${MIRROR_FREEZE_INITS}") ;;
        freeze) config="${PROBE_DIR}/configs/main_freeze.yaml"
                extra+=("env.init_states_per_task=${MIRROR_FREEZE_INITS}") ;;
        *) die "unknown collection '${which}' (want: main | mirror | freeze)" ;;
    esac

    banner "COLLECT ${which}  (num_envs=${envs})"
    setup_sim_env
    log "config: ${config}"

    python "${PROBE_DIR}/run_collect.py" \
        --config "${config}" \
        --set "model.model_path=${CKPT_PATH}" \
              "model.unnorm_key=${UNNORM_KEY}" \
              "env.task_suite_name=${TASK_SUITE}" \
              "env.num_envs=${envs}" \
              ${extra[@]+"${extra[@]}"}

    local run_id
    run_id=$(awk -F': *' '/^run_id:/ {print $2; exit}' "${config}" | tr -d '"'"'"'"')
    log "collection finished; run dir: ${DATA_ROOT}/${run_id}"
    [ -f "${DATA_ROOT}/${run_id}/calls.parquet" ] \
        || die "no calls.parquet at ${DATA_ROOT}/${run_id} -- collection produced nothing"
    du -sh "${DATA_ROOT}/${run_id}"
    banner "COLLECT ${which} DONE"
}

# ---------------------------------------------------------------------------
# archive -- Doctrine 3, step 1.  Frames are excluded: they are the bulk and
# no probe reads them.
# ---------------------------------------------------------------------------
cmd_archive() {
    banner "ARCHIVE (Doctrine 3)"
    mkdir -p "${ARCHIVE_DIR}"
    local stamp; stamp=$(date +%Y%m%d-%H%M%S)
    local tarball="${ARCHIVE_DIR}/efference-runs-${stamp}.tar"

    local runs=()
    while IFS= read -r d; do runs+=("$(basename "$d")"); done < <(
        find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d \
             -not -name 'smoke*' | sort )
    [ ${#runs[@]} -gt 0 ] || die "no collection runs found under ${DATA_ROOT}"
    log "archiving: ${runs[*]}"

    # --exclude frames: the probes read calls.parquet + hidden/*.npz only.
    tar -C "${DATA_ROOT}" --exclude='frames' -cf "${tarball}" "${runs[@]}"
    local sha; sha=$(sha256sum "${tarball}" | cut -d' ' -f1)
    echo "${sha}  $(basename "${tarball}")" > "${tarball}.sha256"

    log "archive: ${tarball}"
    log "size:    $(du -h "${tarball}" | cut -f1)"
    log "sha256:  ${sha}"

    banner "DOWNLOAD NOW -- DO NOT RELEASE THE BOX YET"
    cat <<EOF
From the laptop:

    scp <user>@<box>:${tarball}        .
    scp <user>@<box>:${tarball}.sha256 .

Then verify + gate locally, still with the box running:

    bash experiments/efference_probe/rental_runbook.sh verify \\
        $(basename "${tarball}") ${sha}

Release the rental only after that prints ALL GATES PASSED.
Doctrine 3: never release before local sanity passes; never pay for idle GPU
during analysis.
EOF
}

# ---------------------------------------------------------------------------
# verify -- Doctrine 3, step 2.  Runs on the laptop.  No GPU, no LIBERO, no torch.
#
# Gates are the V2.2 operative set: plumbing passed with zero mismatch counts,
# the indexing check passed, and P0 in [0.45, 0.55].  The old "P2 >= 0.9" gate
# is OBSOLETE -- for the mean-preserving `swap` transform a linear P2 sits at
# chance by design, so it cannot gate anything.
# ---------------------------------------------------------------------------
cmd_verify() {
    local tarball="${1:?usage: verify <archive.tar> <expected-sha256>}"
    local expected="${2:?usage: verify <archive.tar> <expected-sha256>}"
    banner "VERIFY (Doctrine 3, on the laptop)"

    log "checksum"
    local actual; actual=$(sha256sum "${tarball}" | cut -d' ' -f1)
    [ "${actual}" = "${expected}" ] || die "CHECKSUM MISMATCH
     expected ${expected}
     actual   ${actual}
     The transfer is corrupt. Re-download before releasing the box."
    log "sha256 matches"

    local dest="${DATA_ROOT}"
    mkdir -p "${dest}"
    log "extracting into ${dest}"
    tar -xf "${tarball}" -C "${dest}"

    local failed=0
    while IFS= read -r run; do
        banner "GATES: $(basename "${run}")"
        python "${PROBE_DIR}/run_probes.py" \
            --run "${run}" --stage pilot --probes P0 --no-figures --no-controls \
            || { warn "probe run failed for ${run}"; failed=1; continue; }

        python - <<PY || failed=1
import json, sys
from pathlib import Path
import pandas as pd

run = Path("${run}")
out = run / "analysis"
ok = True

plumbing = json.loads((out / "plumbing.json").read_text(encoding="utf-8"))
zero_keys = [k for k in plumbing if k.startswith("n_") and (
    "mismatch" in k or "missing" in k or "unknown" in k or "discontinu" in k)]
print(f"plumbing passed={plumbing.get('passed')}")
for k in sorted(zero_keys):
    print(f"  {k} = {plumbing[k]}")
    if plumbing[k] != 0:
        print(f"  FAIL: {k} must be 0"); ok = False
if not plumbing.get("passed"):
    print("FAIL: plumbing did not pass"); ok = False

manifest_path = run / "manifest.json"
if manifest_path.is_file():
    verification = json.loads(manifest_path.read_text(encoding="utf-8")).get("verification", {})
    print(f"indexing check passed={verification.get('passed')} "
          f"max_abs_logprob_diff={verification.get('max_abs_logprob_diff')}")
    if not verification.get("passed"):
        print("FAIL: indexing check did not pass"); ok = False
else:
    print("FAIL: no manifest.json in the archive"); ok = False

results_path = out / "probe_results.csv"
results = pd.read_csv(results_path) if results_path.is_file() else None
if results is None:
    print(f"FAIL: no {results_path}"); ok = False
else:
    p0 = results[results["name"] == "P0"]["balanced_acc_mean"]
    print(f"P0 cells={len(p0)} mean={p0.mean():.3f} min={p0.min():.3f} max={p0.max():.3f}")
    # P0 is the shuffled-label floor: outside [0.45, 0.55] the labels or the
    # splits leak, and every rung above it is untrustworthy.
    if not (0.45 <= p0.mean() <= 0.55):
        print("FAIL: P0 mean outside [0.45, 0.55]"); ok = False

sys.exit(0 if ok else 1)
PY
    done < <(find "${dest}" -mindepth 1 -maxdepth 1 -type d -not -name 'smoke*' | sort)

    if [ "${failed}" -ne 0 ]; then
        die "GATES FAILED -- keep the box alive and diagnose; the data may need re-collecting"
    fi
    banner "ALL GATES PASSED -- safe to release the rental now"
}

cmd_all() {
    cmd_preflight; cmd_bootstrap; cmd_checkpoint; cmd_smoke
    cmd_collect main; cmd_archive
}

case "${1:-}" in
    preflight)  cmd_preflight ;;
    bootstrap)  cmd_bootstrap ;;
    checkpoint) cmd_checkpoint ;;
    smoke)      cmd_smoke ;;
    collect)    shift; cmd_collect "${@:-main}" ;;
    archive)    cmd_archive ;;
    verify)     shift; cmd_verify "$@" ;;
    all)        cmd_all ;;
    *)
        sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
        exit 1
        ;;
esac
