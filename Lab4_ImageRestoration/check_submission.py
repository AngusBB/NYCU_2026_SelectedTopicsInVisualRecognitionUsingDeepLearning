"""Validate the HW4 pred.npz submission format."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from data import natural_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Path to pred.npz or submission.zip.")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/test/degraded"),
    )
    return parser.parse_args()


def resolve_npz_path(path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if path.suffix == ".npz":
        return path, None
    if path.suffix != ".zip":
        raise ValueError("Submission must be a .npz or .zip file.")
    temp_dir = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if "pred.npz" not in names:
            raise ValueError("Zip must contain pred.npz at the archive root.")
        extra = [name for name in names if name != "pred.npz"]
        if extra:
            raise ValueError(f"Zip contains extra files: {extra[:5]}")
        archive.extract("pred.npz", temp_dir.name)
    return Path(temp_dir.name) / "pred.npz", temp_dir


def expected_shapes(test_dir: Path) -> dict[str, tuple[int, int]]:
    shapes = {}
    for path in sorted(test_dir.glob("*.png"), key=natural_key):
        with Image.open(path) as image:
            width, height = image.convert("RGB").size
        shapes[path.name] = (height, width)
    return shapes


def main() -> None:
    args = parse_args()
    npz_path, temp_dir = resolve_npz_path(args.submission)
    try:
        data = np.load(npz_path)
        expected = expected_shapes(args.test_dir)
        expected_keys = set(expected)
        actual_keys = set(data.keys())
        missing = sorted(expected_keys - actual_keys, key=natural_key)
        extra = sorted(actual_keys - expected_keys, key=natural_key)
        if missing:
            raise ValueError(f"Missing keys: {missing[:10]}")
        if extra:
            raise ValueError(f"Unexpected keys: {extra[:10]}")

        for name in sorted(expected, key=natural_key):
            array = data[name]
            height, width = expected[name]
            if array.shape != (3, height, width):
                raise ValueError(
                    f"{name}: expected shape {(3, height, width)}, got {array.shape}"
                )
            if array.dtype != np.uint8:
                raise ValueError(f"{name}: expected uint8, got {array.dtype}")
            if array.min() < 0 or array.max() > 255:
                raise ValueError(f"{name}: values outside [0, 255]")
        print(f"OK: {len(expected)} predictions match the HW4 submission format.")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
