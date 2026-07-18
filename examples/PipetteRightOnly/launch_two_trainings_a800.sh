#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/WM/DiT4DiT
RUN_ROOT=/workspace/WM/DiT4DiT_runs
PYTHON=/dev/shm/conda_envs/dit4dit/bin/python
ACCELERATE_CONFIG=DiT4DiT/config/deepseeds/deepspeed_zero2.yaml
JOINT_GPUS=${JOINT_GPUS:-0,1}
WRIST_GPUS=${WRIST_GPUS:-2,3}

if [[ ! -x "${PYTHON}" ]]; then
  echo "runtime environment is unavailable; run: bash examples/PipetteRightOnly/restore_a800_environment.sh" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/pipette_right_joints_action_dit" "${RUN_ROOT}/pipette_right_wrist_delta_action_dit"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export HF_HOME=/workspace/WM/DiT4DiT_cache/huggingface
export TORCH_HOME=/workspace/WM/DiT4DiT_cache/torch
export CONDA_PKGS_DIRS=/dev/shm/conda_cache/conda-pkgs
export PIP_CACHE_DIR=/dev/shm/conda_cache/pip
export XDG_CACHE_HOME=/dev/shm/conda_cache/xdg
export TMPDIR=/dev/shm/conda_cache/tmp
export TRITON_CACHE_DIR=/dev/shm/conda_cache/triton/dit4dit
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

mkdir -p "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

CUDA_VISIBLE_DEVICES="${JOINT_GPUS}" nohup "${PYTHON}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_machines 1 --num_processes 2 --main_process_port 29631 \
  DiT4DiT/training/train.py \
  --config_yaml DiT4DiT/config/pipette/dit4dit_pipette_right_joints_a800.yaml \
  >"${RUN_ROOT}/pipette_right_joints_action_dit/train.log" 2>&1 &
echo $! >"${RUN_ROOT}/pipette_right_joints_action_dit/launcher.pid"

CUDA_VISIBLE_DEVICES="${WRIST_GPUS}" nohup "${PYTHON}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_machines 1 --num_processes 2 --main_process_port 29632 \
  DiT4DiT/training/train.py \
  --config_yaml DiT4DiT/config/pipette/dit4dit_pipette_right_wrist_delta_a800.yaml \
  >"${RUN_ROOT}/pipette_right_wrist_delta_action_dit/train.log" 2>&1 &
echo $! >"${RUN_ROOT}/pipette_right_wrist_delta_action_dit/launcher.pid"

echo "joint training: GPUs ${JOINT_GPUS}"
echo "wrist-delta training: GPUs ${WRIST_GPUS}"
