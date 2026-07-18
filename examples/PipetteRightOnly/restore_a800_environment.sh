#!/usr/bin/env bash
set -euo pipefail

HELPER=/workspace/conda_envs/.tools/a800_conda_env.py
RUNTIME=/dev/shm/conda_envs/dit4dit

if [[ ! -f "${HELPER}" ]]; then
  echo "A800 Conda helper not found: ${HELPER}" >&2
  exit 1
fi

python "${HELPER}" status
if [[ ! -x "${RUNTIME}/bin/python" ]]; then
  python "${HELPER}" restore dit4dit
fi
python "${HELPER}" status
"${RUNTIME}/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
