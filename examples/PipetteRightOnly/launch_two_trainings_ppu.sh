#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/WM/DiT4DiT}
RUN_ROOT=${RUN_ROOT:-/workspace/WM/DiT4DiT_runs}
DATA_ROOT=${DATA_ROOT:-/workspace/WM/DiT4DiT_data/derived}
BASE_MODEL=${BASE_MODEL:-/workspace/WM/DiT4DiT_weights/Cosmos-Predict2.5-2B}
PYTHON=${PYTHON:-/workspace/WM/dit4dit_env/bin/python}
PPU_ENV_SETUP=${PPU_ENV_SETUP:-/usr/local/PPU_SDK/envsetup.sh}
ACCELERATE_CONFIG=DiT4DiT/config/deepseeds/deepspeed_zero2.yaml
JOINT_GPUS=${JOINT_GPUS:-0,1}
WRIST_GPUS=${WRIST_GPUS:-2,3}

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
if [[ ! -d "${DATA_ROOT}/pick_up_pipette_right_joints" ]]; then
  echo "Joint-action dataset is unavailable under: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}/pick_up_pipette_right_wrist_delta" ]]; then
  echo "Wrist-delta dataset is unavailable under: ${DATA_ROOT}" >&2
  exit 1
fi

# shellcheck disable=SC1090
set +u
source "${PPU_ENV_SETUP}"
set -u
export UMD_PLATFORM_TYPE=1
export HGGC_DRIVER_CANDIDATE=UMD

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=${WANDB_MODE:-offline}
export HF_HOME=${HF_HOME:-/workspace/WM/DiT4DiT_cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/workspace/WM/DiT4DiT_cache/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/workspace/WM/DiT4DiT_cache/xdg}
export TMPDIR=${TMPDIR:-/workspace/WM/DiT4DiT_cache/tmp}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/workspace/WM/DiT4DiT_cache/triton}
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

mkdir -p \
  "${RUN_ROOT}/pipette_right_joints_action_dit" \
  "${RUN_ROOT}/pipette_right_wrist_delta_action_dit" \
  "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${TRITON_CACHE_DIR}"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${JOINT_GPUS}" nohup "${PYTHON}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_machines 1 --num_processes 2 --main_process_port 29631 \
  DiT4DiT/training/train.py \
  --config_yaml DiT4DiT/config/pipette/dit4dit_pipette_right_joints.yaml \
  --run_root_dir "${RUN_ROOT}" \
  --framework.cosmos25.base_model "${BASE_MODEL}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  >"${RUN_ROOT}/pipette_right_joints_action_dit/train.log" 2>&1 &
echo $! >"${RUN_ROOT}/pipette_right_joints_action_dit/launcher.pid"

CUDA_VISIBLE_DEVICES="${WRIST_GPUS}" nohup "${PYTHON}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_machines 1 --num_processes 2 --main_process_port 29632 \
  DiT4DiT/training/train.py \
  --config_yaml DiT4DiT/config/pipette/dit4dit_pipette_right_wrist_delta.yaml \
  --run_root_dir "${RUN_ROOT}" \
  --framework.cosmos25.base_model "${BASE_MODEL}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  >"${RUN_ROOT}/pipette_right_wrist_delta_action_dit/train.log" 2>&1 &
echo $! >"${RUN_ROOT}/pipette_right_wrist_delta_action_dit/launcher.pid"

echo "joint training: PPU devices ${JOINT_GPUS}"
echo "wrist-delta training: PPU devices ${WRIST_GPUS}"
