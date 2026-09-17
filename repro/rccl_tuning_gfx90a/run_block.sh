#!/bin/bash
#SBATCH --job-name=rccl-tune-block
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-gpu=60G
#SBATCH --time=01:30:00
#SBATCH --output=rccl-tune-%A_%a-%j.out
#SBATCH --error=rccl-tune-%A_%a-%j.err

# One Slurm allocation is one block. Never compare across allocations except
# through the sentinel that opens and closes this one.
#
#   STAGE=0 REPLICATE=1 sbatch repro/rccl_tuning_gfx90a/run_block.sh
#   STAGE=2 REPLICATE=1 BLOCK=1 sbatch --partition=standard-g repro/rccl_tuning_gfx90a/run_block.sh
#
# Normally driven by submit_stage.sh, which submits a whole stage as a job array
# and prints the pre-registration text first. Submitting by hand is fine for a
# single block; submitting a stage by hand defeats the randomisation.
#
# Billed to project_462000131. Override with: sbatch --account=<other> ...

set -euo pipefail

STAGE="${STAGE:-0}"
REPLICATE="${REPLICATE:-${SLURM_ARRAY_TASK_ID:-1}}"
BLOCK="${BLOCK:-1}"
SEED="${SEED:-0}"
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
PROFILE_ARGS="${PROFILE_ARGS:-}"
# Stage 1 turns this on. Measurement slots must leave it at WARN: per-rank INFO
# logging from 32 ranks measurably changes the timings it is meant to explain.
SLOT_DEBUG="${SLOT_DEBUG:-0}"
REAP_SETTLE_S="${REAP_SETTLE_S:-10}"
# Per-slot wall limit, in minutes. Without it one hung slot consumes the whole
# allocation and takes every slot after it down with it -- the block is then
# INCOMPLETE and the GPU-hours are gone. RCCL hangs are the documented failure mode
# on this machine (laifs-container-recipes#44), so a slot that stops making progress
# is expected, not hypothetical. Slurm kills just the step; the loop continues and
# the slot is recorded with its non-zero exit code.
SLOT_TIMEOUT_MIN="${SLOT_TIMEOUT_MIN:-10}"

# The LAIFS mount stalls rather than failing, so probe it before committing the
# allocation to anything.
LAIFS_PROBE_TIMEOUT_S="${LAIFS_PROBE_TIMEOUT_S:-300}"
for path in /appl/local/laifs/modules "$(dirname "${CONTAINER}")"; do
  echo "probing ${path} (up to ${LAIFS_PROBE_TIMEOUT_S}s)..."
  if ! timeout "${LAIFS_PROBE_TIMEOUT_S}" ls "${path}" >/dev/null 2>&1; then
    echo "ERROR: ${path} is unreachable (LAIFS mount stalled). Aborting." >&2
    exit 1
  fi
done

module load Local-LAIF lumi-aif-singularity-bindings

WORKDIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUNTIME_DIR="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm_runtime/${SLURM_JOB_ID}"
REL="repro/rccl_tuning_gfx90a"
RESULTS_HOST="${WORKDIR}/${REL}/results/job_${SLURM_JOB_ID}"
RESULTS_CONT="/work/${REL}/results/job_${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}" "${RESULTS_HOST}"

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")
IN_CONTAINER=(singularity run "${BIND_ARGS[@]}" "${CONTAINER}")

echo "stage=${STAGE} replicate=${REPLICATE} block=${BLOCK} seed=${SEED}"
echo "nodes=${SLURM_JOB_NUM_NODES} world=${SLURM_NPROCS} partition=${SLURM_JOB_PARTITION}"
echo "nodelist=${SLURM_JOB_NODELIST}"
echo "results=${RESULTS_HOST}"

# Persistent per-user MIOpen cache. /tmp is node-local, so this only pays off when
# Slurm reuses nodes, but it costs nothing and removes one source of slot-1 bias.
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"
srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
  mkdir -p "${MIOPEN_CUSTOM_CACHE_DIR}" "${MIOPEN_USER_DB_PATH}"

# The design is generated inside the container so the study depends on exactly one
# python, and written out before anything runs so the realised permutation is on
# disk even if the allocation dies halfway.
"${IN_CONTAINER[@]}" python3 "/work/${REL}/designs.py" \
  --stage "${STAGE}" --replicate "${REPLICATE}" --seed "${SEED}" --block "${BLOCK}" \
  > "${RESULTS_HOST}/block.json"

"${IN_CONTAINER[@]}" python3 -c '
import json, sys
block = json.load(open(sys.argv[1]))
for slot in block["slots"]:
    kv = " ".join(f"{k}={v}" for k, v in sorted(slot["env"].items())) or "-"
    # "-" rather than "": tab is IFS whitespace, so bash collapses consecutive tabs
    # and an empty field silently shifts every field after it.
    print("\t".join([slot["name"], slot["role"], str(slot["position"]),
                     slot.get("srun_flags") or "-", kv,
                     json.dumps(slot["env"], sort_keys=True)]))
' "${RESULTS_HOST}/block.json" > "${RESULTS_HOST}/slots.tsv"

MANAGED_VARS="$("${IN_CONTAINER[@]}" python3 -c '
import sys; sys.path.insert(0, "/work/'"${REL}"'")
import designs; print(" ".join(designs.MANAGED_VARS))')"
echo "managed vars: ${MANAGED_VARS}"

export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export WORLD_SIZE="${SLURM_NPROCS}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-8}"
export RESULTS_DIR="${RESULTS_CONT}"
export CONTAINER
export BLOCK_STAGE="${STAGE}" BLOCK_REPLICATE="${REPLICATE}" BLOCK_ID="${BLOCK}" BLOCK_SEED="${SEED}"
export BLOCK_PERMUTATION="$(cut -f1 "${RESULTS_HOST}/slots.tsv" | paste -sd, -)"
echo "permutation: ${BLOCK_PERMUTATION}"

STATUS_FILE="${RESULTS_HOST}/slot_status.tsv"
printf 'name\trole\tposition\texit_code\twall_s\tnote\n' > "${STATUS_FILE}"

while IFS=$'\t' read -r name role position flags kv env_json <&3; do
  [ -n "${name}" ] || continue
  echo
  echo "=== slot ${position}: ${name} (${role}) ==="

  # A variable must never survive into the next slot. Unsetting the whole managed
  # set is the only way to be sure -- the ancestor unset one variable by name and
  # that only worked because it swept exactly one knob.
  for var in ${MANAGED_VARS}; do unset "${var}" || true; done
  [ "${flags}" = "-" ] && flags=""
  if [ "${kv}" != "-" ]; then
    for pair in ${kv}; do export "${pair?}"; done
  fi

  # A stale port from a slot that died mid-rendezvous will hang the next one.
  export MASTER_PORT="$(( 20000 + (SLURM_JOB_ID % 10000) + RANDOM % 1000 ))"
  export BLOCK_POSITION="${position}" BLOCK_ROLE="${role}"
  export SLOT_ENV_JSON="${env_json}"
  export SLOT_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ "${SLOT_DEBUG}" = "1" ]; then
    # INFO from every rank, written to Lustre, took job 22115267 from ~17 s a slot to
    # 263 s and timed the whole block out -- the same way job 21843529 died in the bug
    # study. Log to node-local /tmp and drop GRAPH and NET, which are the verbose ones
    # and assert nothing that verify_effect.py reads.
    export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING
    export FI_LOG_LEVEL=warn FI_LOG_PROV=cxi
    # NCCL_DEBUG_FILE is set per rank inside in_container_slot.sh: NCCL substitutes
    # only %h and %p, so a %d here is written literally and nothing finds the file.
  else
    export NCCL_DEBUG=WARN
  fi

  slot_start=${SECONDS}
  # --kill-on-bad-exit=0 so one rank dying does not tear down the block; the
  # reaping step below is what stops the survivors poisoning the next slot.
  set +e
  # PROFILE_ARGS arrives via --export=ALL but is only a shell variable here until
  # it is exported, and in_container_slot.sh reads it from the environment.
  export SLOT_NAME="${name}" REL_DIR="${REL}" SLOT_DEBUG PROFILE_ARGS
  srun --kill-on-bad-exit=0 --time="${SLOT_TIMEOUT_MIN}" ${flags} \
    "${IN_CONTAINER[@]}" "/work/${REL}/in_container_slot.sh"
  rc=$?
  set -e
  slot_wall=$(( SECONDS - slot_start ))
  note="ok"
  [ "${rc}" -ne 0 ] && note="failed"
  [ "${slot_wall}" -ge $(( SLOT_TIMEOUT_MIN * 60 - 15 )) ] && note="TIMED_OUT"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${name}" "${role}" "${position}" "${rc}" \
    "${slot_wall}" "${note}" >> "${STATUS_FILE}"
  echo "slot ${name} exit=${rc} wall=${slot_wall}s ${note}"

  # Ranks left blocked inside RCCL hold their GCDs and counterfeit an intermittent
  # bug in whatever runs next. Job 21790393 produced 11 invalid rows without this.
  srun --overlap --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
    bash -c 'pkill -f "[c]ollective_profile\.py" || true' >/dev/null 2>&1 || true
  sleep "${REAP_SETTLE_S}"
done 3< "${RESULTS_HOST}/slots.tsv"

# Job 22114824 ran 1 of 10 slots and still exited 0, because srun ate the loop's
# stdin. A block that silently measures a fraction of its design is worse than one
# that fails, so the count is checked rather than assumed.
expected=$(wc -l < "${RESULTS_HOST}/slots.tsv")
completed=$(( $(wc -l < "${STATUS_FILE}") - 1 ))
echo
echo "=== slots: ${completed}/${expected} completed ==="
if [ "${completed}" -ne "${expected}" ]; then
  echo "ERROR: the block ran ${completed} of ${expected} slots. Its results are not a" >&2
  echo "       block and must not be pooled with others. Discard this job id." >&2
  touch "${RESULTS_HOST}/INCOMPLETE_BLOCK"
fi

echo
echo "=== analysis ==="
"${IN_CONTAINER[@]}" python3 "/work/${REL}/analyze.py" \
  --results-dir "${RESULTS_CONT}" || true

echo
echo "slot status:"
cat "${STATUS_FILE}"
