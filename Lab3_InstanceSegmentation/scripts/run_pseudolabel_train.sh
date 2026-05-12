#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
CONDA_ENV="${CONDA_ENV:-segmentation}"
CONFIG="${CONFIG:-$ROOT/configs/htc_swin_b_copypaste_all_random-erasing_flipd4_pseudolabel.py}"
WORK_DIR="${WORK_DIR:-$ROOT/work_dirs/htc_swin_b_copypaste_all_random-erasing_flipd4_pseudolabel_ft}"
LOG="${LOG:-$ROOT/logs/pseudolabel_ft_train.log}"
GPUS="${GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PORT="${PORT:-29517}"
PSEUDO_SUBMISSION="${PSEUDO_SUBMISSION:-$ROOT/sweep_outputs/ensembles/ensembles_37-1p0_34-1p0_nms0p60.zip}"
PSEUDO_SCORE_THR="${PSEUDO_SCORE_THR:-0.20}"
PSEUDO_MAX_PER_IMG="${PSEUDO_MAX_PER_IMG:-1000}"
PSEUDO_ANN="${PSEUDO_ANN:-$ROOT/data/processed/annotations/instances_train_all_plus_pseudo_ens37_34_thr0p20.json}"

cd "$ROOT"
mkdir -p "$WORK_DIR" "$(dirname "$LOG")" "$(dirname "$PSEUDO_ANN")"

if [[ -n "$CONDA_ENV" ]]; then
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  elif command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
fi

python scripts/build_pseudo_label_coco.py \
  --submission "$PSEUDO_SUBMISSION" \
  --score-thr "$PSEUDO_SCORE_THR" \
  --max-per-img "$PSEUDO_MAX_PER_IMG" \
  --output "$PSEUDO_ANN"

echo "Pseudo-label self-finetuning"
echo "  config: $CONFIG"
echo "  pseudo_ann: $PSEUDO_ANN"
echo "  pseudo_score_thr: $PSEUDO_SCORE_THR"
echo "  base checkpoint: work_dirs/htc_swin_b_copypaste_all_random-erasing_flipd4/epoch_24.pth"
echo "  epochs: 8"
echo "  work_dir: $WORK_DIR"
echo "  gpus: $GPUS ($CUDA_VISIBLE_DEVICES)"
echo "  log: $LOG"

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PORT="$PORT" \
bash external/mmdetection/tools/dist_train.sh \
  "$CONFIG" "$GPUS" \
  --work-dir "$WORK_DIR" 2>&1 | tee "$LOG"
