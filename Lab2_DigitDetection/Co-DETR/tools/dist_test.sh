#!/usr/bin/env bash

set -euo pipefail

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
python -m torch.distributed.launch --nproc_per_node="${GPUS}" --master_port="${PORT}" \
    "${SCRIPT_DIR}/test.py" "${CONFIG}" "${CHECKPOINT}" --launcher pytorch "${@:4}"
