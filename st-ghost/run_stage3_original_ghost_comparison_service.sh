#!/usr/bin/env bash
set -uo pipefail

stage3_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
stage3_wrapper="$stage3_root/st-ghost/run_stage3_original_ghost_comparison_service.sh"
stage3_target="$stage3_root/st-ghost/run_stage3_original_ghost_comparison.py"
stage3_config="$stage3_root/st-ghost/configs/stage3_original_ghost_comparison.json"
stage3_session="${STGHOST_STAGE3_SESSION:-stghost-stage3-original-ghost-comparison}"
# The experiment initializes its own output directory and rejects foreign files.
stage3_state="${STGHOST_STAGE3_STATE:-$stage3_root/logs/stghost-stage3-original-ghost-comparison-service}"
stage3_log="${STGHOST_STAGE3_LOG:-$stage3_root/logs/stghost-stage3-original-ghost-comparison.log}"
stage3_status="$stage3_state/exit_status"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${STGHOST_GPU_INDEX:-0}}"

screen_alive() {
    screen -list 2>/dev/null | grep -q "[.]${stage3_session}[[:space:]]"
}

mkdir -p "$stage3_state" "$(dirname -- "$stage3_log")" || exit 1
cd "$stage3_root" || exit 1

if [[ "${1:-}" == "--screen-worker" ]]; then
    date --iso-8601=seconds > "$stage3_state/started_at"
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$stage3_root/.venv-stage1/bin/python" -u "$stage3_target" \
        --config "$stage3_config" 2>&1 | tee -a "$stage3_log"
    stage3_code=${PIPESTATUS[0]}
    date --iso-8601=seconds > "$stage3_state/finished_at"
    printf '%s\n' "$stage3_code" > "$stage3_status.tmp"
    mv "$stage3_status.tmp" "$stage3_status"
    exit "$stage3_code"
fi

exec 9>"$stage3_state/supervisor.lock"
if ! flock -n 9; then
    printf 'Another Stage 3 supervisor is running.\n' >&2
    exit 75
fi
if screen_alive; then
    printf 'Screen session %s is already running.\n' "$stage3_session" >&2
    exit 75
fi

stage3_gpu_index="${STGHOST_GPU_INDEX:-0}"
stage3_min_free_mib="${STGHOST_MIN_FREE_MIB:-46081}"
if [[ ! "$stage3_min_free_mib" =~ ^[0-9]+$ ]]; then
    printf 'Invalid STGHOST_MIN_FREE_MIB: %s\n' "$stage3_min_free_mib" >&2
    exit 1
fi
if ! stage3_free_memory=$(nvidia-smi -i "$stage3_gpu_index" --query-gpu=memory.free --format=csv,noheader,nounits); then
    printf 'Cannot query GPU %s free memory; launch postponed.\n' "$stage3_gpu_index" >&2
    exit 75
fi
stage3_free_memory=${stage3_free_memory//[[:space:]]/}
if [[ ! "$stage3_free_memory" =~ ^[0-9]+$ ]] || (( stage3_free_memory < stage3_min_free_mib )); then
    printf 'GPU %s has %s MiB free; strictly more than %s MiB is required.\n' \
        "$stage3_gpu_index" "$stage3_free_memory" "$((stage3_min_free_mib - 1))" >&2
    exit 75
fi

rm -f "$stage3_status" "$stage3_state/finished_at"
screen -DmS "$stage3_session" /bin/bash "$stage3_wrapper" --screen-worker \
    >> "$stage3_state/screen.log" 2>&1 &
stage3_screen_pid=$!

for ((stage3_attempt = 0; stage3_attempt < 30; stage3_attempt++)); do
    if screen_alive || [[ -f "$stage3_status" ]]; then
        break
    fi
    if ! kill -0 "$stage3_screen_pid" 2>/dev/null; then
        break
    fi
    sleep 1
done
while screen_alive; do
    sleep 5
done
wait "$stage3_screen_pid"
stage3_screen_code=$?

if [[ -r "$stage3_status" && "$(<"$stage3_status")" == "0" ]]; then
    exit 0
fi
printf 'Stage 3 screen/worker exited without success (screen status %s); see %s and %s.\n' \
    "$stage3_screen_code" "$stage3_state/screen.log" "$stage3_log" >&2
exit 1
