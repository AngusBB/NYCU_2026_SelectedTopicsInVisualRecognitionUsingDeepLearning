#!/usr/bin/env bash

set -euo pipefail

CONFIG=$1
GPUS=$2
WORKDIR=$3

PORT=${PORT:-29500}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
python -m torch.distributed.launch --nproc_per_node="${GPUS}" --master_port="${PORT}" \
    "${SCRIPT_DIR}/train.py" "${CONFIG}" --launcher pytorch "${@:4}" --work-dir "${WORKDIR}"
