"""HTC Swin-B with Copy-Paste augmentation for cell segmentation ablation."""

import sys

sys.path.insert(0, '.')

_base_ = './htc_swin_b.py'

custom_imports = dict(
    imports=['custom_transforms'],
    allow_failed_imports=False)

data_root = './data/processed/'
classes = ('class1', 'class2', 'class3', 'class4')
metainfo = dict(classes=classes)
backend_args = None

copy_paste_load_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,
         with_seg=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (480, 1333), (512, 1333), (544, 1333),
                        (576, 1333), (608, 1333), (640, 1333),
                        (672, 1333), (704, 1333), (736, 1333),
                        (768, 1333), (800, 1333)
                    ],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (480, 1333), (512, 1333), (544, 1333),
                        (576, 1333), (608, 1333), (640, 1333),
                        (672, 1333), (704, 1333), (736, 1333),
                        (768, 1333), (800, 1333)
                    ],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='FilterAnnotations',
        min_gt_bbox_wh=(1e-2, 1e-2),
        min_gt_mask_area=4,
        by_mask=True),
]

copy_paste_pipeline = [
    dict(
        type='ShapeAwareCopyPaste',
        max_num_pasted=80,
        bbox_occluded_thr=10,
        mask_occluded_thr=64,
        selected=True,
        size_divisor=32),
    dict(type='RebuildSemanticSeg', ignore_index=255),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/instances_train.json',
            data_prefix=dict(img='images/train/', seg='semantic/train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=copy_paste_load_pipeline,
            backend_args=backend_args),
        pipeline=copy_paste_pipeline,
        max_refetch=30))

test_evaluator = dict(
    outfile_prefix=(
        './work_dirs/'
        'htc_swin_b_copypaste/test'))

default_hooks = dict(
    checkpoint=dict(save_best='coco/segm_mAP', rule='greater'))

work_dir = './work_dirs/htc_swin_b_copypaste'
