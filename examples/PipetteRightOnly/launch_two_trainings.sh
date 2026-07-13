#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/nvme02/DiT4DiT
RUN_ROOT=/home/nvme02/DiT4DiT_runs
CONDA=/home/nvme01/miniconda3/bin/conda
ACCELERATE_CONFIG=DiT4DiT/config/deepseeds/deepspeed_zero2.yaml

mkdir -p "${RUN_ROOT}/pipette_right_joints_action_dit" "${RUN_ROOT}/pipette_right_wrist_delta_action_dit"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=online
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

CUDA_VISIBLE_DEVICES=0,3 nohup "${CONDA}" run -n dit4dit accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_machines 1 --num_processes 2 --main_process_port 29631 \
  DiT4DiT/training/train.py \
  --config_yaml DiT4DiT/config/pipette/dit4dit_pipette_right_joints.yaml \
  >"${RUN_ROOT}/pipette_right_joints_action_dit/train.log" 2>&1 &
echo $! >"${RUN_ROOT}/pipette_right_joints_action_dit/launcher.pid"

CUDA_VISIBLE_DEVICES=2,6 nohup "${CONDA}" run -n dit4dit accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_machines 1 --num_processes 2 --main_process_port 29632 \
  DiT4DiT/training/train.py \
  --config_yaml DiT4DiT/config/pipette/dit4dit_pipette_right_wrist_delta.yaml \
  >"${RUN_ROOT}/pipette_right_wrist_delta_action_dit/train.log" 2>&1 &
echo $! >"${RUN_ROOT}/pipette_right_wrist_delta_action_dit/launcher.pid"

echo "joint training: GPUs 0,3"
echo "wrist-delta training: GPUs 2,6"
