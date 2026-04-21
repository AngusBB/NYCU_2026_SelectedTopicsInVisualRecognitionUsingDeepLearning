#!/usr/bin/env python3
"""Create a super-resolved COCO dataset and scale annotations.

This script is intentionally backend-agnostic:

* ``--backend bicubic`` is fast and useful for verifying the resized dataset
  and annotation geometry.
* ``--backend tair`` first resizes each image to the requested output geometry,
  then restores that resized image with TAIR/TeReDiff.

For labeled splits, the generated COCO JSON files keep the original image ids
and category ids, but update image width/height, bbox coordinates, and bbox
areas.  For the test split, no annotation JSON is read or written.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("nycu-hw2-data"))
    parser.add_argument("--output-root", type=Path, default=Path("nycu-hw2-data_tair_1024"))
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument(
        "--resize-mode",
        choices=["long-edge", "short-edge", "square-pad", "square-stretch"],
        default="long-edge",
        help=(
            "long-edge preserves aspect ratio and sets the longer side to target-size. "
            "square-pad preserves aspect ratio and pads to target-size x target-size. "
            "square-stretch distorts aspect ratio and is mainly for ablation."
        ),
    )
    parser.add_argument(
        "--pad-position",
        choices=["top-left", "center"],
        default="top-left",
        help="Only used with --resize-mode square-pad.",
    )
    parser.add_argument("--pad-value", type=int, default=0)
    parser.add_argument("--backend", choices=["bicubic", "tair"], default="bicubic")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--jpg-quality", type=int, default=95)

    parser.add_argument("--tair-root", type=Path, default=Path("TAIR"))
    parser.add_argument("--tair-config", type=Path, default=Path("TAIR/configs/val/val_terediff.yaml"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--sampler-type", type=str, default="spaced")
    parser.add_argument("--pos-prompt", type=str, default="")
    parser.add_argument("--neg-prompt", type=str, default="")
    parser.add_argument("--noise-aug", type=int, default=0)
    parser.add_argument("--start-point-type", choices=["random", "cond"], default="random")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled TAIR inference.")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-stride", type=int, default=256)
    return parser.parse_args()


@dataclass(frozen=True)
class ResizeMeta:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    output_width: int
    output_height: int
    scale_x: float
    scale_y: float
    offset_x: float = 0.0
    offset_y: float = 0.0


def compute_resize_meta(width: int, height: int, args: argparse.Namespace) -> ResizeMeta:
    target = args.target_size
    if args.resize_mode == "square-stretch":
        resized_width = target
        resized_height = target
        output_width = target
        output_height = target
        offset_x = 0
        offset_y = 0
    else:
        if args.resize_mode in {"long-edge", "square-pad"}:
            scale = target / max(width, height)
        elif args.resize_mode == "short-edge":
            scale = target / min(width, height)
        else:
            raise ValueError(args.resize_mode)

        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        if args.resize_mode == "square-pad":
            output_width = target
            output_height = target
            if args.pad_position == "center":
                offset_x = (target - resized_width) / 2.0
                offset_y = (target - resized_height) / 2.0
            else:
                offset_x = 0.0
                offset_y = 0.0
        else:
            output_width = resized_width
            output_height = resized_height
            offset_x = 0.0
            offset_y = 0.0

    return ResizeMeta(
        original_width=width,
        original_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        output_width=output_width,
        output_height=output_height,
        scale_x=resized_width / width,
        scale_y=resized_height / height,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def resize_input_image(image: Image.Image, meta: ResizeMeta, args: argparse.Namespace) -> Image.Image:
    image = image.convert("RGB")
    image = image.resize((meta.resized_width, meta.resized_height), Image.Resampling.BICUBIC)
    if args.resize_mode != "square-pad":
        return image

    canvas = Image.new("RGB", (meta.output_width, meta.output_height), (args.pad_value,) * 3)
    canvas.paste(image, (int(round(meta.offset_x)), int(round(meta.offset_y))))
    return canvas


def scale_bbox(bbox: list[float], meta: ResizeMeta) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    x = x * meta.scale_x + meta.offset_x
    y = y * meta.scale_y + meta.offset_y
    w = w * meta.scale_x
    h = h * meta.scale_y
    x = min(max(x, 0.0), float(meta.output_width))
    y = min(max(y, 0.0), float(meta.output_height))
    w = min(max(w, 0.0), float(meta.output_width) - x)
    h = min(max(h, 0.0), float(meta.output_height) - y)
    return [x, y, w, h]


def scale_segmentation(segmentation: Any, meta: ResizeMeta) -> Any:
    if not isinstance(segmentation, list):
        return segmentation

    scaled = []
    for poly in segmentation:
        if not isinstance(poly, list):
            scaled.append(poly)
            continue
        out = []
        for i, value in enumerate(poly):
            if i % 2 == 0:
                out.append(float(value) * meta.scale_x + meta.offset_x)
            else:
                out.append(float(value) * meta.scale_y + meta.offset_y)
        scaled.append(out)
    return scaled


class BicubicRestorer:
    def restore(self, image: Image.Image) -> Image.Image:
        return image


class TairRestorer:
    def __init__(self, args: argparse.Namespace) -> None:
        tair_root = args.tair_root.resolve()
        if not tair_root.is_dir():
            raise FileNotFoundError(f"TAIR repo does not exist: {tair_root}")
        sys.path.insert(0, str(tair_root))

        try:
            import torch
            from omegaconf import OmegaConf
            from terediff.model import ControlLDM, Diffusion, SwinIR
            from terediff.pipeline import SwinIRPipeline
            from terediff.utils.common import instantiate_from_config
        except Exception as exc:  # pragma: no cover - setup dependent
            raise RuntimeError(
                "Failed to import TAIR dependencies. Install TAIR requirements, "
                "detectron2, and testr in the active environment first."
            ) from exc

        self.torch = torch
        self.args = args
        self.device = torch.device(args.device)
        cfg = OmegaConf.load(args.tair_config)

        cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
        sd_path = Path(str(cfg.train.sd_path))
        if not sd_path.is_absolute():
            sd_path = tair_root / sd_path
        sd = torch.load(sd_path, map_location="cpu")["state_dict"]
        cldm.load_pretrained_sd(sd)

        resume_path = Path(str(cfg.train.resume))
        if not resume_path.is_absolute():
            resume_path = tair_root / resume_path
        if resume_path.is_file():
            cldm.load_controlnet_from_ckpt(torch.load(resume_path, map_location="cpu"))
        else:
            cldm.load_controlnet_from_unet()

        resume_ckpt = cfg.exp_args.get("resume_ckpt_dir", None)
        if resume_ckpt:
            resume_ckpt = Path(str(resume_ckpt))
            if not resume_ckpt.is_absolute():
                resume_ckpt = tair_root / resume_ckpt
            if resume_ckpt.is_file():
                ckpt = torch.load(resume_ckpt, map_location="cpu")
                if "cldm" in ckpt:
                    cldm.load_state_dict(ckpt["cldm"], strict=False)
                elif "state_dict" in ckpt:
                    cldm.load_state_dict(ckpt["state_dict"], strict=False)
                else:
                    cldm.load_state_dict(ckpt, strict=False)
            else:
                raise FileNotFoundError(
                    f"cfg.exp_args.resume_ckpt_dir points to a missing checkpoint: {resume_ckpt}"
                )

        swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
        swinir_path = Path(str(cfg.train.swinir_path))
        if not swinir_path.is_absolute():
            swinir_path = tair_root / swinir_path
        sd = torch.load(swinir_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
        swinir.load_state_dict(sd, strict=True)

        diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
        self.pipeline = SwinIRPipeline(
            cleaner=swinir.eval().to(self.device),
            cldm=cldm.eval().to(self.device),
            diffusion=diffusion.to(self.device),
            cond_fn=None,
            device=str(self.device),
        )

    def restore(self, image: Image.Image) -> Image.Image:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)[None, ...]
        with self.torch.inference_mode():
            out = self.pipeline.run(
                arr,
                steps=self.args.steps,
                strength=self.args.strength,
                cleaner_tiled=self.args.tiled,
                cleaner_tile_size=self.args.tile_size,
                cleaner_tile_stride=self.args.tile_stride,
                vae_encoder_tiled=self.args.tiled,
                vae_encoder_tile_size=self.args.tile_size,
                vae_decoder_tiled=self.args.tiled,
                vae_decoder_tile_size=self.args.tile_size,
                cldm_tiled=self.args.tiled,
                cldm_tile_size=self.args.tile_size,
                cldm_tile_stride=self.args.tile_stride,
                pos_prompt=self.args.pos_prompt,
                neg_prompt=self.args.neg_prompt,
                cfg_scale=self.args.cfg_scale,
                start_point_type=self.args.start_point_type,
                sampler_type=self.args.sampler_type,
                noise_aug=self.args.noise_aug,
                rescale_cfg=False,
                s_churn=0.0,
                s_tmin=0.0,
                s_tmax=0.0,
                s_noise=1.0,
                eta=0.0,
                order=1,
            )[0]
        return Image.fromarray(out)


def build_restorer(args: argparse.Namespace):
    if args.backend == "bicubic":
        return BicubicRestorer()
    return TairRestorer(args)


def iter_images(coco: dict[str, Any], max_images: int | None) -> Iterable[dict[str, Any]]:
    images = sorted(coco["images"], key=lambda item: int(item["id"]))
    if max_images is not None:
        images = images[:max_images]
    return images


def infer_image_id(path: Path, fallback: int) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return fallback


def build_image_only_coco(image_dir: Path) -> dict[str, Any]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = []
    for fallback, path in enumerate(sorted(p for p in image_dir.iterdir() if p.suffix.lower() in suffixes), start=0):
        with Image.open(path) as image:
            width, height = image.size
        images.append(
            {
                "id": infer_image_id(path, fallback),
                "file_name": path.name,
                "height": int(height),
                "width": int(width),
            }
        )
    return {"images": images, "annotations": [], "categories": []}


def write_image(image: Image.Image, path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_format == "jpg":
        image.save(path.with_suffix(".jpg"), quality=args.jpg_quality)
    else:
        image.save(path.with_suffix(".png"))


def output_file_name(file_name: str, args: argparse.Namespace) -> str:
    return str(Path(file_name).with_suffix(f".{args.save_format}"))


def process_split(split: str, restorer, args: argparse.Namespace) -> None:
    image_dir = args.data_root / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    write_annotations = split != "test"
    if write_annotations:
        ann_path = args.data_root / f"{split}.json"
        if not ann_path.is_file():
            raise FileNotFoundError(f"Missing annotation file for labeled split: {ann_path}")
        with ann_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
    else:
        print(f"[{split}] processing image-only split without reading or writing annotation JSON")
        coco = build_image_only_coco(image_dir)

    out_coco = copy.deepcopy(coco)
    out_image_dir = args.output_root / split
    args.output_root.mkdir(parents=True, exist_ok=True)

    metas_by_image_id: dict[int, ResizeMeta] = {}
    selected_ids = {int(img["id"]) for img in iter_images(coco, args.max_images)}
    images_by_id = {int(img["id"]): img for img in out_coco["images"]}

    selected_images = list(iter_images(coco, args.max_images))
    for idx, image_info in enumerate(selected_images, start=1):
        image_id = int(image_info["id"])
        input_path = image_dir / image_info["file_name"]
        out_name = output_file_name(image_info["file_name"], args)
        output_path = out_image_dir / out_name

        meta = compute_resize_meta(int(image_info["width"]), int(image_info["height"]), args)
        metas_by_image_id[image_id] = meta
        images_by_id[image_id]["width"] = int(meta.output_width)
        images_by_id[image_id]["height"] = int(meta.output_height)
        images_by_id[image_id]["file_name"] = out_name

        if args.skip_existing and output_path.is_file():
            if idx % 100 == 0 or idx == len(selected_images):
                print(f"[{split}] {idx}/{len(selected_images)} skipped/existing")
            continue

        with Image.open(input_path) as image:
            resized = resize_input_image(image, meta, args)
            restored = restorer.restore(resized)
        write_image(restored, output_path, args)

        if idx % 50 == 0 or idx == len(selected_images):
            print(f"[{split}] {idx}/{len(selected_images)} images")

    if args.max_images is not None:
        out_coco["images"] = [img for img in out_coco["images"] if int(img["id"]) in selected_ids]

    if write_annotations:
        out_annotations = []
        for ann in out_coco.get("annotations", []):
            image_id = int(ann["image_id"])
            if image_id not in metas_by_image_id:
                if args.max_images is None:
                    raise KeyError(f"Missing resize metadata for image_id={image_id}")
                continue
            meta = metas_by_image_id[image_id]
            new_ann = ann
            new_ann["bbox"] = scale_bbox(new_ann["bbox"], meta)
            new_ann["area"] = float(new_ann["bbox"][2] * new_ann["bbox"][3])
            if "segmentation" in new_ann:
                new_ann["segmentation"] = scale_segmentation(new_ann["segmentation"], meta)
            out_annotations.append(new_ann)
        if "annotations" in out_coco:
            out_coco["annotations"] = out_annotations

        out_ann_path = args.output_root / f"{split}.json"
        with out_ann_path.open("w", encoding="utf-8") as handle:
            json.dump(out_coco, handle)
        print(f"[{split}] wrote {out_ann_path} and {out_image_dir}")
    else:
        print(f"[{split}] wrote images to {out_image_dir}")


def main() -> None:
    args = parse_args()
    if args.target_size <= 0:
        raise ValueError("--target-size must be positive")
    if not 0 <= args.pad_value <= 255:
        raise ValueError("--pad-value must be in [0, 255]")

    restorer = build_restorer(args)
    for split in args.splits:
        process_split(split, restorer, args)


if __name__ == "__main__":
    main()
