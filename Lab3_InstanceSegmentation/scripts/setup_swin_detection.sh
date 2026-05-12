#!/usr/bin/env bash
set -euo pipefail

# Blackwell-compatible OpenMMLab setup for the existing `segmentation`
# environment. Run after `conda activate segmentation`.

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CHECKPOINT_DIR="${1:-./checkpoints}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "nvcc not found at ${CUDA_HOME}/bin/nvcc" >&2
  echo "Set CUDA_HOME to a CUDA toolkit that matches the PyTorch CUDA runtime." >&2
  exit 1
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export MMCV_WITH_OPS="${MMCV_WITH_OPS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-12}"

pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

pip install -U openmim
pip install 'setuptools<70' wheel ninja cython 'numpy<2'
pip install mmengine 'mmcv==2.1.0' --no-build-isolation -v
pip install 'numpy<2' 'opencv-python<4.12' 'mmdet==3.3.0' pycocotools tifffile

mkdir -p "${CHECKPOINT_DIR}"
if [[ ! -f "${CHECKPOINT_DIR}/swin_base_patch4_window7_224_22k.pth" ]]; then
  wget -O "${CHECKPOINT_DIR}/swin_base_patch4_window7_224_22k.pth" \
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
fi

python - <<'PY'
import torch
from mmcv.ops import nms, roi_align

print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
    print('capability', torch.cuda.get_device_capability(0))

boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]], device='cuda')
scores = torch.tensor([0.9, 0.8], device='cuda')
print('nms keep', nms(boxes, scores, 0.5)[1].detach().cpu().tolist())

feat = torch.randn(1, 1, 8, 8, device='cuda')
rois = torch.tensor([[0., 0., 0., 7., 7.]], device='cuda')
print('roi_align', tuple(roi_align(feat, rois, (2, 2), 1.0, 0, 'avg', True).shape))
PY

echo "OpenMMLab environment is ready. Checkpoint directory: ${CHECKPOINT_DIR}"
