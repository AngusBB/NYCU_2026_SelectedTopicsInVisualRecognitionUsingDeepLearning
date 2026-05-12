"""Project-local MMDetection transforms."""

from __future__ import annotations

import numpy as np
import mmcv
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms.transforms import CopyPaste
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks


@TRANSFORMS.register_module()
class ShapeAwareCopyPaste(CopyPaste):
    """Copy-Paste after padding source/destination to the same shape."""

    def __init__(self,
                 size_divisor: int = 32,
                 img_pad_val: int = 0,
                 mask_pad_val: int = 0,
                 seg_pad_val: int = 255,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_divisor = size_divisor
        self.img_pad_val = img_pad_val
        self.mask_pad_val = mask_pad_val
        self.seg_pad_val = seg_pad_val

    def _copy_paste(self, dst_results: dict, src_results: dict) -> dict:
        self._pad_to_same_shape(dst_results, src_results)
        return super()._copy_paste(dst_results, src_results)

    def _pad_to_same_shape(self, dst_results: dict, src_results: dict) -> None:
        target_h = max(dst_results['img'].shape[0], src_results['img'].shape[0])
        target_w = max(dst_results['img'].shape[1], src_results['img'].shape[1])

        if self.size_divisor:
            target_h = int(np.ceil(target_h / self.size_divisor) *
                           self.size_divisor)
            target_w = int(np.ceil(target_w / self.size_divisor) *
                           self.size_divisor)

        for results in (dst_results, src_results):
            self._pad_one(results, target_h, target_w)

    def _pad_one(self, results: dict, target_h: int, target_w: int) -> None:
        pad_shape = (target_h, target_w)
        if results['img'].shape[:2] != pad_shape:
            pad_val = self.img_pad_val
            if isinstance(pad_val, int) and results['img'].ndim == 3:
                pad_val = tuple(pad_val for _ in range(results['img'].shape[2]))
            results['img'] = mmcv.impad(
                results['img'],
                shape=pad_shape,
                pad_val=pad_val)

        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.impad(
                results['gt_seg_map'],
                shape=pad_shape,
                pad_val=self.seg_pad_val)

        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].pad(
                pad_shape,
                pad_val=self.mask_pad_val)

        results['img_shape'] = results['img'].shape[:2]
        results['pad_shape'] = results['img'].shape
        results['pad_fixed_size'] = None
        results['pad_size_divisor'] = self.size_divisor


@TRANSFORMS.register_module()
class RareClassCopyPaste(ShapeAwareCopyPaste):
    """Shape-aware Copy-Paste biased toward rare classes."""

    def __init__(self,
                 rare_classes: tuple[int, ...] = (2, 3),
                 rare_image_prob: float = 0.75,
                 rare_class_weight: float = 8.0,
                 min_num_pasted: int = 20,
                 min_rare_pasted: int = 8,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.rare_classes = tuple(int(label) for label in rare_classes)
        self.rare_image_prob = float(rare_image_prob)
        self.rare_class_weight = float(rare_class_weight)
        self.min_num_pasted = int(min_num_pasted)
        self.min_rare_pasted = int(min_rare_pasted)
        self._rare_image_indices: list[int] | None = None

    @cache_randomness
    def get_indexes(self, dataset) -> int:
        rare_indices = self._get_rare_image_indices(dataset)
        if rare_indices and np.random.rand() < self.rare_image_prob:
            return int(np.random.choice(rare_indices))
        return int(np.random.randint(0, len(dataset)))

    def _get_rare_image_indices(self, dataset) -> list[int]:
        if self._rare_image_indices is not None:
            return self._rare_image_indices

        rare_set = set(self.rare_classes)
        indices = []
        for index in range(len(dataset)):
            data_info = dataset.get_data_info(index)
            labels = [
                int(instance.get('bbox_label', -1))
                for instance in data_info.get('instances', [])
                if not instance.get('ignore_flag', 0)
            ]
            if any(label in rare_set for label in labels):
                indices.append(index)

        self._rare_image_indices = indices
        return indices

    def _select_object(self, results: dict) -> dict:
        bboxes = results['gt_bboxes']
        labels = results['gt_bboxes_labels']
        masks = self.get_gt_masks(results)
        ignore_flags = results['gt_ignore_flags']

        selected_inds = self._get_rare_weighted_inds(labels)

        results['gt_bboxes'] = bboxes[selected_inds]
        results['gt_bboxes_labels'] = labels[selected_inds]
        results['gt_masks'] = masks[selected_inds]
        results['gt_ignore_flags'] = ignore_flags[selected_inds]
        return results

    @cache_randomness
    def _get_rare_weighted_inds(self, labels: np.ndarray) -> np.ndarray:
        num_bboxes = len(labels)
        if num_bboxes == 0:
            return np.asarray([], dtype=np.int64)

        max_num_pasted = min(num_bboxes, self.max_num_pasted)
        min_num_pasted = min(max(0, self.min_num_pasted), max_num_pasted)
        if max_num_pasted <= min_num_pasted:
            num_pasted = max_num_pasted
        else:
            num_pasted = np.random.randint(min_num_pasted, max_num_pasted + 1)

        if num_pasted == 0:
            return np.asarray([], dtype=np.int64)

        labels = np.asarray(labels)
        rare_mask = np.isin(labels, self.rare_classes)
        rare_inds = np.flatnonzero(rare_mask)
        common_inds = np.flatnonzero(~rare_mask)

        selected: list[int] = []
        if len(rare_inds) > 0:
            rare_quota = min(len(rare_inds), num_pasted, self.min_rare_pasted)
            if rare_quota > 0:
                selected.extend(
                    np.random.choice(
                        rare_inds, size=rare_quota, replace=False).tolist())

        remaining = num_pasted - len(selected)
        if remaining > 0:
            available = np.setdiff1d(
                np.arange(num_bboxes), np.asarray(selected, dtype=np.int64),
                assume_unique=False)
            weights = np.ones(len(available), dtype=np.float64)
            weights[np.isin(labels[available], self.rare_classes)] = (
                self.rare_class_weight)
            weights = weights / weights.sum()
            selected.extend(
                np.random.choice(
                    available, size=remaining, replace=False,
                    p=weights).tolist())

        return np.asarray(selected, dtype=np.int64)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(rare_classes={self.rare_classes}, '
            f'rare_image_prob={self.rare_image_prob}, '
            f'rare_class_weight={self.rare_class_weight}, '
            f'min_num_pasted={self.min_num_pasted}, '
            f'min_rare_pasted={self.min_rare_pasted}, '
            f'max_num_pasted={self.max_num_pasted})')


@TRANSFORMS.register_module()
class RebuildSemanticSeg(BaseTransform):
    """Rebuild HTC semantic supervision from instance masks.

    CopyPaste updates instance masks but does not update ``gt_seg_map``. HTC's
    semantic head still consumes that map, so rebuild it after mixed-image
    augmentation to keep the auxiliary semantic loss aligned with the masks.
    """

    def __init__(self, ignore_index: int = 255) -> None:
        self.ignore_index = ignore_index

    def transform(self, results: dict) -> dict:
        if 'gt_masks' not in results or 'gt_bboxes_labels' not in results:
            return results

        height, width = results['img'].shape[:2]
        gt_seg_map = np.full((height, width), self.ignore_index, dtype=np.uint8)
        masks = results['gt_masks'].to_ndarray().astype(bool)
        labels = results['gt_bboxes_labels']

        for label, mask in zip(labels, masks):
            gt_seg_map[mask] = int(label)

        results['gt_seg_map'] = gt_seg_map
        results['ignore_index'] = self.ignore_index
        seg_fields = results.setdefault('seg_fields', [])
        if 'gt_seg_map' not in seg_fields:
            seg_fields.append('gt_seg_map')
        return results


@TRANSFORMS.register_module()
class RandomD4(BaseTransform):
    """Random square-symmetry geometry transform for mask supervision.

    The operation is applied to the image, instance masks, bboxes, and semantic
    map. Boxes are recomputed from the transformed masks to avoid subtle
    off-by-one errors for 90-degree rotations on rectangular images.
    """

    _VALID_OPS = (
        'identity', 'rot90', 'rot180', 'rot270',
        'hflip', 'hflip_rot90', 'hflip_rot180', 'hflip_rot270')

    def __init__(self,
                 prob: float = 1.0,
                 transforms: tuple[str, ...] = _VALID_OPS) -> None:
        self.prob = prob
        self.transforms = tuple(transforms)
        invalid = set(self.transforms) - set(self._VALID_OPS)
        if invalid:
            raise ValueError(f'Invalid D4 transforms: {sorted(invalid)}')

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            op = 'identity'
        else:
            op = str(np.random.choice(self.transforms))

        if op == 'identity':
            results['d4_transform'] = op
            return results

        results['img'] = self._apply_array(results['img'], op)
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = self._apply_array(results['gt_seg_map'], op)

        if results.get('gt_masks', None) is not None:
            masks = results['gt_masks'].to_ndarray()
            masks = self._apply_masks(masks, op)
            height, width = masks.shape[1:]
            results['gt_masks'] = BitmapMasks(masks, height, width)
            self._recompute_bboxes_from_masks(results, masks)

        results['img_shape'] = results['img'].shape[:2]
        results['pad_shape'] = results['img'].shape
        results['d4_transform'] = op
        return results

    def _apply_array(self, array: np.ndarray, op: str) -> np.ndarray:
        output = array
        if op.startswith('hflip'):
            output = np.flip(output, axis=1)
        if 'rot90' in op:
            output = np.rot90(output, k=1, axes=(0, 1))
        elif 'rot180' in op:
            output = np.rot90(output, k=2, axes=(0, 1))
        elif 'rot270' in op:
            output = np.rot90(output, k=3, axes=(0, 1))
        return np.ascontiguousarray(output)

    def _apply_masks(self, masks: np.ndarray, op: str) -> np.ndarray:
        output = masks
        if op.startswith('hflip'):
            output = np.flip(output, axis=2)
        if 'rot90' in op:
            output = np.rot90(output, k=1, axes=(1, 2))
        elif 'rot180' in op:
            output = np.rot90(output, k=2, axes=(1, 2))
        elif 'rot270' in op:
            output = np.rot90(output, k=3, axes=(1, 2))
        return np.ascontiguousarray(output.astype(np.uint8))

    def _recompute_bboxes_from_masks(self, results: dict,
                                     masks: np.ndarray) -> None:
        boxes = []
        for mask in masks:
            ys, xs = np.nonzero(mask)
            if len(xs) == 0 or len(ys) == 0:
                boxes.append([0.0, 0.0, 0.0, 0.0])
            else:
                boxes.append([
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max() + 1),
                    float(ys.max() + 1),
                ])
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        original_boxes = results.get('gt_bboxes')
        if hasattr(original_boxes, 'tensor'):
            results['gt_bboxes'] = type(original_boxes)(
                boxes,
                dtype=original_boxes.tensor.dtype,
                device=original_boxes.tensor.device)
        else:
            results['gt_bboxes'] = HorizontalBoxes(boxes)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(prob={self.prob}, transforms={self.transforms})'
