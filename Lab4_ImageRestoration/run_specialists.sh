#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-data}
GPUS=${GPUS:-0}
NPROC=${NPROC:-1}
EPOCHS=${EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-6}
PATCH_SIZE=${PATCH_SIZE:-0}
LR=${LR:-2e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
AMP=${AMP:-bf16}
NUM_WORKERS=${NUM_WORKERS:-8}
VAL_RATIO=${VAL_RATIO:-0.1}
SEED=${SEED:-42}
RAIN_RUN=${RAIN_RUN:-rain_full256_ep200}
SNOW_RUN=${SNOW_RUN:-snow_full256_ep200}
CLASSIFIER_CSV=${CLASSIFIER_CSV:-runs/weather_classifier_resnet18_rgbres_ep20/test_weather_classifier_predictions.csv}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/specialists_routed_tta}
RUN_RAIN=${RUN_RAIN:-1}
RUN_SNOW=${RUN_SNOW:-1}
RUN_PREDICT=${RUN_PREDICT:-1}
TTA=${TTA:-1}

train_kind() {
  local kind=$1
  local run_name=$2
  echo "==> Training ${kind} specialist: ${run_name}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NPROC}" \
    train.py \
    --data-root "${DATA_ROOT}" \
    --run-name "${run_name}" \
    --kind-filter "${kind}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --patch-size "${PATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --val-ratio "${VAL_RATIO}" \
    --seed "${SEED}" \
    --amp "${AMP}" \
    --num-workers "${NUM_WORKERS}"
}

if [[ "${RUN_RAIN}" == "1" ]]; then
  train_kind rain "${RAIN_RUN}"
fi

if [[ "${RUN_SNOW}" == "1" ]]; then
  train_kind snow "${SNOW_RUN}"
fi

if [[ "${RUN_PREDICT}" == "1" ]]; then
  predict_args=()
  if [[ "${TTA}" == "1" ]]; then
    predict_args+=(--tta)
  fi

  echo "==> Routing specialist predictions into ${OUTPUT_DIR}"
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "${PYTHON}" predict_promptir_specialists.py \
    --rain-checkpoint "runs/${RAIN_RUN}/best.pt" \
    --snow-checkpoint "runs/${SNOW_RUN}/best.pt" \
    --classifier-csv "${CLASSIFIER_CSV}" \
    --input-dir "${DATA_ROOT}/test/degraded" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 8 \
    --num-workers "${NUM_WORKERS}" \
    --amp "${AMP}" \
    "${predict_args[@]}"

  "${PYTHON}" check_submission.py "${OUTPUT_DIR}/pred.npz" \
    --test-dir "${DATA_ROOT}/test/degraded"
fi
