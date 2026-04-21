#!/usr/bin/env bash
set -euo pipefail

MMCV_VERSION="${MMCV_VERSION:-1.7.0}"
BUILD_ROOT="${BUILD_ROOT:-${TMPDIR:-/tmp}}"
BUILD_DIR="${BUILD_DIR:-${BUILD_ROOT}/mmcv-${MMCV_VERSION}-modern-build}"
MAX_JOBS="${MAX_JOBS:-1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"

if ! command -v python >/dev/null 2>&1; then
  echo "python is not available in PATH" >&2
  exit 1
fi

if [ -z "${TORCH_CUDA_ARCH_LIST}" ]; then
  TORCH_CUDA_ARCH_LIST="$(
    python - <<'PY'
try:
    import torch
except Exception:
    print("")
    raise SystemExit(0)

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}.{minor}")
else:
    print("")
PY
  )"
fi

if [ -z "${TORCH_CUDA_ARCH_LIST}" ]; then
  echo "Could not detect a CUDA compute capability automatically." >&2
  echo "Please set TORCH_CUDA_ARCH_LIST explicitly, for example 8.6 for RTX A6000." >&2
  exit 1
fi

echo "Building MMCV ${MMCV_VERSION} for the active Python environment"
echo "Build directory: ${BUILD_DIR}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "MAX_JOBS=${MAX_JOBS}"

if [ -e "${BUILD_DIR}" ]; then
  echo "Removing existing build directory ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

git clone --depth 1 --branch "v${MMCV_VERSION}" https://github.com/open-mmlab/mmcv.git "${BUILD_DIR}"

python - <<'PY' "${BUILD_DIR}"
from pathlib import Path
import sys

root = Path(sys.argv[1])

setup_path = root / "setup.py"
setup_text = setup_path.read_text()
setup_text = setup_text.replace("-std=c++14", "-std=c++17")
setup_path.write_text(setup_text)

config_path = root / "mmcv" / "utils" / "config.py"
config_text = config_path.read_text()
config_text = config_text.replace(
    "text, _ = FormatCode(text, style_config=yapf_style, verify=True)",
    "text, _ = FormatCode(text, style_config=yapf_style)",
)
config_path.write_text(config_text)

parallel_path = root / "mmcv" / "parallel" / "_functions.py"
parallel_text = parallel_path.read_text()
old = "streams = [_get_stream(device) for device in target_gpus]"
new = (
    "streams = [\n"
    "            _get_stream(device if isinstance(device, torch.device) else torch.device('cuda', device))\n"
    "            for device in target_gpus\n"
    "        ]"
)
if old not in parallel_text:
    raise SystemExit(f"Could not find expected stream helper in {parallel_path}")
parallel_text = parallel_text.replace(old, new)
parallel_path.write_text(parallel_text)

distributed_path = root / "mmcv" / "parallel" / "distributed.py"
distributed_text = distributed_path.read_text()
old = (
    "        module_to_run = self._replicated_tensor_module if \\\n"
    "            self._use_replicated_tensor_module else self.module\n"
)
new = (
    "        module_to_run = self.module\n"
    "        if getattr(self, '_use_replicated_tensor_module', False):\n"
    "            module_to_run = getattr(self, '_replicated_tensor_module', self.module)\n"
)
if old not in distributed_text:
    raise SystemExit(f"Could not find expected DDP forward helper in {distributed_path}")
distributed_text = distributed_text.replace(old, new)
distributed_path.write_text(distributed_text)
PY

pushd "${BUILD_DIR}" >/dev/null
python -m pip install -U pip "setuptools<82" wheel
python -m pip install -r requirements/runtime.txt

TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
MMCV_WITH_OPS=1 \
MAX_JOBS="${MAX_JOBS}" \
python setup.py build_ext

TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
MMCV_WITH_OPS=1 \
MAX_JOBS="${MAX_JOBS}" \
python setup.py install
popd >/dev/null

python - <<'PY' "${BUILD_DIR}"
from pathlib import Path
import importlib.util
import shutil
import sys

root = Path(sys.argv[1])
spec = importlib.util.find_spec("mmcv")
if spec is None or spec.submodule_search_locations is None:
    raise SystemExit("mmcv is not importable after installation")

site_mmcv = Path(list(spec.submodule_search_locations)[0])
built_exts = list((root / "build").rglob("_ext*.so"))
if not built_exts:
    raise SystemExit("Could not find compiled mmcv _ext shared library")

target = site_mmcv / built_exts[0].name
if not target.exists():
    shutil.copy2(built_exts[0], target)
    print(f"Copied {built_exts[0]} -> {target}")
else:
    print(f"Found existing extension at {target}")
PY

python - <<'PY'
import mmcv
import torch
from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401

print("mmcv:", mmcv.__version__)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
PY
