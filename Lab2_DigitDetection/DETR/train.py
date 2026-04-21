"""Train a ResNet-50 DETR model for digit detection."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from src.config import deep_copy_config, load_config, save_config
from src.data.coco import COCODetectionDataset
from src.data.transforms import build_eval_transforms, build_train_transforms
from src.engine.evaluator import evaluate_coco
from src.engine.trainer import build_optimizer, build_scheduler, train_one_epoch
from src.models.detr import build_detr
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.distributed import destroy_process_group, init_distributed_mode, is_main_process
from src.utils.ema import ModelEMA
from src.utils.misc import collate_fn, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/detr_r50.yaml")
    parser.add_argument("--train-image-dir", nargs="+", default=["nycu-hw2-data/train"])
    parser.add_argument("--train-ann", nargs="+", default=["nycu-hw2-data/train.json"])
    parser.add_argument("--val-image-dir", type=str, default="nycu-hw2-data/valid")
    parser.add_argument("--val-ann", type=str, default="nycu-hw2-data/valid.json")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--disable-tensorboard", action="store_true")
    return parser.parse_args()


def build_train_dataset(config, image_dirs, ann_files):
    if len(image_dirs) != len(ann_files):
        raise ValueError("--train-image-dir and --train-ann must have the same length.")
    data_cfg = config["data"]
    mosaic_center_ratio_range = tuple(
        float(value) for value in data_cfg.get("mosaic_center_ratio_range", (0.4, 0.6))
    )
    datasets = [
        COCODetectionDataset(
            image_dir=image_dir,
            annotation_file=ann_file,
            transforms=build_train_transforms(config),
            category_ids=config["model"]["category_ids"],
            mosaic_probability=float(data_cfg.get("mosaic_probability", 0.0)),
            mosaic_size=int(data_cfg["mosaic_size"]) if data_cfg.get("mosaic_size") is not None else None,
            mosaic_center_ratio_range=mosaic_center_ratio_range,
            mosaic_min_box_size=float(data_cfg.get("mosaic_min_box_size", 2.0)),
            mixup_probability=float(data_cfg.get("mixup_probability", 0.0)),
            mixup_alpha=float(data_cfg.get("mixup_alpha", 8.0)),
            mixup_min_box_size=float(data_cfg.get("mixup_min_box_size", 2.0)),
        )
        for image_dir, ann_file in zip(image_dirs, ann_files)
    ]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def create_summary_writer(
    args: argparse.Namespace,
    config,
    output_dir: Path,
    world_size: int,
):
    if args.disable_tensorboard or not is_main_process():
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print(
            "TensorBoard logging is disabled because the `tensorboard` package is not installed. "
            "Install it with `pip install tensorboard` or `pip install -r requirements.txt`."
        )
        return None

    log_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else output_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("run/config", f"```json\n{json.dumps(config, indent=2)}\n```", 0)
    writer.add_text(
        "run/command",
        f"`{' '.join(shlex.quote(arg) for arg in sys.argv)}`",
        0,
    )
    writer.add_scalar("run/world_size", world_size, 0)
    writer.add_scalar("run/per_gpu_batch_size", int(config["data"]["batch_size"]), 0)
    writer.add_scalar(
        "run/global_batch_size",
        int(config["data"]["batch_size"]) * world_size * int(config["train"]["grad_accum_steps"]),
        0,
    )
    return writer


def log_metrics(writer, prefix: str, metrics, step: int) -> None:
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}", value, step)


def should_log_training_images(config: Dict, epoch: int) -> bool:
    image_every = int(config["train"].get("tensorboard_image_every", 5))
    return image_every > 0 and (epoch == 1 or epoch % image_every == 0)


def _render_training_sample(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    class_names: List[str],
    mean: List[float],
    std: List[float],
) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype).view(-1, 1, 1)
    rendered = (image.detach().cpu() * std_tensor + mean_tensor).clamp(0.0, 1.0)
    rendered = (rendered * 255.0).round().to(torch.uint8)

    boxes = target.get("boxes")
    labels = target.get("labels")
    if boxes is None or labels is None or boxes.numel() == 0:
        return rendered.float() / 255.0

    from torchvision.utils import draw_bounding_boxes

    boxes = boxes.detach().cpu().clone()
    labels = labels.detach().cpu().clone()
    _, height, width = rendered.shape
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, max(width - 1, 0))
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, max(height - 1, 0))
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not valid.any():
        return rendered.float() / 255.0

    text_labels = [
        class_names[int(label)]
        if 0 <= int(label) < len(class_names)
        else str(int(label))
        for label in labels[valid].tolist()
    ]
    rendered = draw_bounding_boxes(
        rendered,
        boxes=boxes[valid].round().to(torch.int64),
        labels=text_labels,
        colors="lime",
        width=2,
    )
    return rendered.float() / 255.0


def log_training_images(
    writer,
    preview_samples: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    config: Dict,
    epoch: int,
) -> None:
    if writer is None or not preview_samples:
        return

    from torchvision.utils import make_grid

    class_names = list(config["model"]["class_names"])
    mean = list(config["data"]["image_mean"])
    std = list(config["data"]["image_std"])
    rendered_list = [
        _render_training_sample(image, target, class_names, mean, std)
        for image, target in preview_samples
    ]
    for index, image in enumerate(rendered_list):
        writer.add_image(f"train/augmented_samples/sample_{index:02d}", image, epoch)

    max_height = max(image.shape[1] for image in rendered_list)
    max_width = max(image.shape[2] for image in rendered_list)
    rendered_images = torch.stack(
        [
            torch.nn.functional.pad(
                image,
                (0, max_width - image.shape[2], 0, max_height - image.shape[1]),
                value=0.0,
            )
            for image in rendered_list
        ]
    )
    writer.add_image("train/augmented_grid", make_grid(rendered_images, nrow=4), epoch)
    writer.add_scalar("train/augmented_sample_count", rendered_images.shape[0], epoch)


def main() -> None:
    args = parse_args()
    dist_state = init_distributed_mode()
    device = dist_state["device"]

    config = deep_copy_config(load_config(args.config))
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.seed is not None:
        config["train"]["seed"] = args.seed

    seed_everything(int(config["train"]["seed"]) + dist_state["rank"])

    if args.output_dir is None:
        args.output_dir = f"outputs/ep{config['train']['epochs']}_bs{config['data']['batch_size']}_lr{config['train']['lr']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        save_config(config, output_dir / f"resolved_config_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.yaml")
    writer = create_summary_writer(args, config, output_dir, dist_state["world_size"])

    train_dataset = build_train_dataset(config, args.train_image_dir, args.train_ann)
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)
        if dist_state["distributed"]
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["data"]["batch_size"]),
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = None
    if args.val_image_dir is not None and args.val_ann is not None:
        val_dataset = COCODetectionDataset(
            image_dir=args.val_image_dir,
            annotation_file=args.val_ann,
            transforms=build_eval_transforms(config),
            category_ids=config["model"]["category_ids"],
        )
        val_sampler = (
            DistributedSampler(val_dataset, shuffle=False)
            if dist_state["distributed"]
            else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config["data"]["batch_size"]),
            sampler=val_sampler,
            shuffle=False,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

    model, criterion, postprocessor = build_detr(config)
    model.to(device)
    criterion.to(device)

    model_without_ddp = model
    ema = ModelEMA(model_without_ddp, decay=float(config["train"]["ema_decay"]))

    if dist_state["distributed"]:
        model = DDP(
            model,
            device_ids=[dist_state["local_rank"]] if device.type == "cuda" else None,
        )

    optimizer = build_optimizer(model_without_ddp, config)
    scheduler = build_scheduler(optimizer, steps_per_epoch=len(train_loader), config=config)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda")

    start_epoch = 1
    best_metric = float("-inf")
    best_ema_metric = float("-inf")

    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            model=model_without_ddp,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            map_location="cpu",
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
        best_ema_metric = float(checkpoint.get("best_ema_metric", best_ema_metric))

    for epoch in range(start_epoch, int(config["train"]["epochs"]) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        preview_callback = None
        if writer is not None and should_log_training_images(config, epoch):
            preview_callback = lambda samples, step_epoch: log_training_images(writer, samples, config, step_epoch)

        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            scaler=scaler,
            config=config,
            ema=ema,
            preview_callback=preview_callback,
            preview_images=int(config["train"].get("tensorboard_num_images", 20)),
        )

        if is_main_process():
            print(f"Train metrics: {train_metrics}")
            if writer is not None:
                log_metrics(writer, "train", train_metrics, epoch)

        eval_every = int(config["train"]["eval_every"])
        should_evaluate = val_loader is not None and epoch % eval_every == 0

        val_metrics = {}
        ema_metrics = {}
        if should_evaluate:
            val_metrics = evaluate_coco(
                model=model,
                criterion=criterion,
                postprocessor=postprocessor,
                data_loader=val_loader,
                device=device,
                score_threshold=float(config["train"]["score_threshold"]),
                max_detections_per_image=int(config["inference"]["max_detections_per_image"]),
                use_amp=bool(config["train"]["amp"]),
            )
            if is_main_process():
                print(f"Validation metrics: {val_metrics}")
                if writer is not None:
                    log_metrics(writer, "val", val_metrics, epoch)

            ema_metrics = evaluate_coco(
                model=ema.module.to(device),
                criterion=criterion,
                postprocessor=postprocessor,
                data_loader=val_loader,
                device=device,
                score_threshold=float(config["train"]["score_threshold"]),
                max_detections_per_image=int(config["inference"]["max_detections_per_image"]),
                use_amp=bool(config["train"]["amp"]),
            )
            if is_main_process():
                print(f"EMA validation metrics: {ema_metrics}")
                if writer is not None:
                    log_metrics(writer, "ema_val", ema_metrics, epoch)

        if is_main_process():
            checkpoint = {
                "epoch": epoch,
                "config": config,
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.state_dict(),
                "best_metric": best_metric,
                "best_ema_metric": best_ema_metric,
            }
            save_checkpoint(checkpoint, output_dir / "last.pth")

            if epoch % int(config["train"]["save_every"]) == 0:
                save_checkpoint(checkpoint, output_dir / f"epoch_{epoch:03d}.pth")

            current_metric = float(val_metrics.get("bbox_mAP", float("-inf")))
            current_ema_metric = float(ema_metrics.get("bbox_mAP", float("-inf")))

            if current_metric > best_metric:
                best_metric = current_metric
                checkpoint["best_metric"] = best_metric
                save_checkpoint(checkpoint, output_dir / "best.pth")

            if current_ema_metric > best_ema_metric:
                best_ema_metric = current_ema_metric
                checkpoint["best_ema_metric"] = best_ema_metric
                save_checkpoint(checkpoint, output_dir / "best_ema.pth")

            if writer is not None:
                writer.add_scalar("best/raw_bbox_mAP", best_metric, epoch)
                writer.add_scalar("best/ema_bbox_mAP", best_ema_metric, epoch)
                writer.flush()

    if is_main_process():
        print(f"Best bbox_mAP: {best_metric:.4f}")
        print(f"Best EMA bbox_mAP: {best_ema_metric:.4f}")
        if writer is not None:
            writer.close()

    destroy_process_group()


if __name__ == "__main__":
    main()
