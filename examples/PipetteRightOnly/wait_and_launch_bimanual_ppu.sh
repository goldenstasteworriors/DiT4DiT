#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/WM/DiT4DiT}
RUN_ROOT=${RUN_ROOT:-/workspace/WM/DiT4DiT_runs}
BIMANUAL_GPUS=${BIMANUAL_GPUS:-10,11,12,13}
POLL_SECONDS=${POLL_SECONDS:-60}
NVIDIA_SMI=${NVIDIA_SMI:-/usr/local/PPU_SDK/CUDA_SDK/bin/nvidia-smi}

IFS=, read -r -a devices <<<"${BIMANUAL_GPUS}"
if [[ ${#devices[@]} -ne 4 ]]; then
  echo "BIMANUAL_GPUS must contain exactly four devices: ${BIMANUAL_GPUS}" >&2
  exit 1
fi

while true; do
  gpu_status=$("${NVIDIA_SMI}" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  free_count=0
  for device in "${devices[@]}"; do
    if awk -F, -v target="${device}" '
      $1 + 0 == target {
        gsub(/ /, "", $2)
        gsub(/ /, "", $3)
        if ($2 + 0 < 1024 && $3 + 0 == 0) found = 1
      }
      END { exit !found }
    ' <<<"${gpu_status}"; then
      ((free_count += 1))
    fi
  done
  if [[ ${free_count} -eq 4 ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

cd "${ROOT}"
RUN_FILTER=bimanual_wrist_delta \
  BIMANUAL_GPUS="${BIMANUAL_GPUS}" \
  RUN_ROOT="${RUN_ROOT}" \
  bash examples/PipetteRightOnly/launch_four_grab_red_bottle_trainings_ppu.sh
