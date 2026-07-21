#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/WM/DiT4DiT}
RUN_ROOT=${RUN_ROOT:-/workspace/WM/DiT4DiT_runs}
DATA_ROOT=${DATA_ROOT:-/workspace/WM/DiT4DiT_data/grab_red_bottle}
BASE_MODEL=${BASE_MODEL:-/workspace/WM/DiT4DiT_weights/Cosmos-Predict2.5-2B}
PYTHON=${PYTHON:-/workspace/WM/dit4dit_env/bin/python}
PPU_ENV_SETUP=${PPU_ENV_SETUP:-/usr/local/PPU_SDK/envsetup.sh}
ACCELERATE_CONFIG=DiT4DiT/config/deepseeds/deepspeed_zero2.yaml
TRAIN_CONFIG=DiT4DiT/config/pipette/dit4dit_grab_red_bottle.yaml
JOINT_GPUS=${JOINT_GPUS:-0,1,2,3}
WRIST_GPUS=${WRIST_GPUS:-4,5,6,7}
TARGET_JOINT_GPUS=${TARGET_JOINT_GPUS:-8,9,10,11}
BIMANUAL_GPUS=${BIMANUAL_GPUS:-12,13,14,15}
RUN_FILTER=${RUN_FILTER:-all}
NUM_PROCESSES=${NUM_PROCESSES:-4}

if [[ ! -x "${PYTHON}" ]]; then
  echo "Project environment is unavailable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${PPU_ENV_SETUP}" ]]; then
  echo "PPU environment setup is unavailable: ${PPU_ENV_SETUP}" >&2
  exit 1
fi
if [[ ! -f "${BASE_MODEL}/model_index.json" ]]; then
  echo "Cosmos backbone is unavailable: ${BASE_MODEL}" >&2
  exit 1
fi
for split in train test; do
  for dataset in \
    pick_up_pipette_right_joints \
    pick_up_pipette_right_wrist_delta \
    pick_up_pipette_right_target_joints \
    pick_up_pipette_bimanual_wrist_delta; do
    if [[ ! -d "${DATA_ROOT}/${split}/${dataset}" ]]; then
      echo "Dataset is unavailable: ${DATA_ROOT}/${split}/${dataset}" >&2
      exit 1
    fi
  done
done

# shellcheck disable=SC1090
set +u
source "${PPU_ENV_SETUP}"
set -u
export UMD_PLATFORM_TYPE=1
export HGGC_DRIVER_CANDIDATE=UMD
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export WANDB_API_KEY=${WANDB_API_KEY:-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}
export WANDB_MODE=${WANDB_MODE:-online}
export HF_HOME=${HF_HOME:-/workspace/WM/DiT4DiT_cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/workspace/WM/DiT4DiT_cache/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/workspace/WM/DiT4DiT_cache/xdg}
export TMPDIR=${TMPDIR:-/workspace/WM/DiT4DiT_cache/tmp}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/workspace/WM/DiT4DiT_cache/triton}
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

mkdir -p "${RUN_ROOT}" "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${TRITON_CACHE_DIR}"
cd "${ROOT}"

start_run() {
  local run_id=$1
  local devices=$2
  local port=$3
  local data_mix=$4
  local dimension=$5
  local seed=$6
  local run_dir="${RUN_ROOT}/${run_id}"
  mkdir -p "${run_dir}"
  if [[ -f "${run_dir}/launcher.pid" ]] && kill -0 "$(<"${run_dir}/launcher.pid")" 2>/dev/null; then
    echo "Run is already active: ${run_id}" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${devices}" nohup "${PYTHON}" -m accelerate.commands.launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_machines 1 --num_processes "${NUM_PROCESSES}" --main_process_port "${port}" \
    DiT4DiT/training/train.py \
    --config_yaml "${TRAIN_CONFIG}" \
    --run_id "${run_id}" \
    --wandb_project "${run_id}" \
    --seed "${seed}" \
    --run_root_dir "${RUN_ROOT}" \
    --framework.cosmos25.base_model "${BASE_MODEL}" \
    --framework.action_model.action_dim "${dimension}" \
    --framework.action_model.state_dim "${dimension}" \
    --datasets.vla_data.data_root_dir "${DATA_ROOT}/train" \
    --datasets.vla_data.data_mix "${data_mix}" \
    --datasets.vla_data.max_state_dim "${dimension}" \
    --datasets.vla_data.max_action_dim "${dimension}" \
    --datasets.vla_test_data.data_root_dir "${DATA_ROOT}/test" \
    --datasets.vla_test_data.data_mix "${data_mix}" \
    --datasets.vla_test_data.max_state_dim "${dimension}" \
    --datasets.vla_test_data.max_action_dim "${dimension}" \
    >"${run_dir}/train.log" 2>&1 &
  echo $! >"${run_dir}/launcher.pid"
  echo "${run_id}: devices=${devices}, processes=${NUM_PROCESSES}, pid=$(<"${run_dir}/launcher.pid")"
}

should_start() {
  [[ "${RUN_FILTER}" == "all" || ",${RUN_FILTER}," == *",$1,"* ]]
}

should_start right_joints && \
  start_run grab_red_bottle_right_joints_action_dit "${JOINT_GPUS}" 29641 pipette_right_joints 16 42
should_start right_wrist_delta && \
  start_run grab_red_bottle_right_wrist_delta_action_dit "${WRIST_GPUS}" 29642 pipette_right_wrist_delta 16 43
should_start right_target_joints && \
  start_run grab_red_bottle_right_target_joints_action_dit "${TARGET_JOINT_GPUS}" 29643 pipette_right_target_joints 16 44
should_start bimanual_wrist_delta && \
  start_run grab_red_bottle_bimanual_wrist_delta_action_dit "${BIMANUAL_GPUS}" 29644 pipette_bimanual_wrist_delta 32 45
