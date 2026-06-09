#!/bin/bash
# DiT4DiT ZMQ Inference Server
# Usage: bash server_zmq.sh

export PYTHONPATH=$(pwd)

python deployment/model_server/server_policy_zmq.py \
    --ckpt_path /path/to/checkpoint/pytorch_model.pt \
    --port 5556 \
    --use_bf16