#!/usr/bin/env bash
set -euo pipefail

ARCHIVE=/workspace/WM/DiT4DiT_env/dit4dit_env.tar.gz
RUNTIME=/dev/shm/dit4dit_env

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "environment archive not found: ${ARCHIVE}" >&2
  exit 1
fi

rm -rf "${RUNTIME}"
mkdir -p "${RUNTIME}"
tar -xzf "${ARCHIVE}" -C "${RUNTIME}"
"${RUNTIME}/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
