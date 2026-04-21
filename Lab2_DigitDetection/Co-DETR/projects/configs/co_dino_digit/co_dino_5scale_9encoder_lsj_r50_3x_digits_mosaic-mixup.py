_base_ = [
    '../co_dino/co_dino_5scale_9encoder_lsj_r50_3x_coco.py',
]

custom_imports = dict(
    imports=[
        'projects.hooks.multi_image_mix_switch_hook',
        'projects.pipelines.mixup_with_prob',
        'projects.pipelines.mosaic_with_affine_flag',
    ],
    allow_failed_imports=False,
)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
num_classes = len(classes)
dataset_type = 'CocoDataset'
data_root = '../nycu-hw2-data/'
image_size = (1024, 1024)
num_last_epochs = 6

model = dict(
    query_head=dict(num_classes=num_classes),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56,
            ),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=12.0,
                ),
                loss_bbox=dict(type='GIoULoss', loss_weight=120.0),
            ),
        )
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=12.0,
            ),
            loss_bbox=dict(type='GIoULoss', loss_weight=24.0),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=12.0,
            ),
        )
    ],
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]

train_pipeline = [
    dict(
        type='MosaicWithAffineFlag',
        img_scale=image_size,
        center_ratio_range=(0.9, 1.1),
        min_bbox_size=2,
        skip_filter=False,
        pad_val=114.0,
        prob=0.2,
    ),
    dict(
        type='ConditionalRandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=0.05,
        scaling_ratio_range=(0.9, 1.1),
        mosaic_border=(-image_size[0] // 2, -image_size[1] // 2),
        plain_border=(0, 0),
        border_val=(114, 114, 114),
        min_bbox_size=2,
        min_area_ratio=0.2,
        max_aspect_ratio=10,
        skip_filter=False,
    ),
    dict(
        type='MixUpWithProb',
        prob=0.2,
        img_scale=image_size,
        ratio_range=(0.98, 1.02),
        flip_ratio=0.2,
        pad_val=114.0,
        min_bbox_size=2,
        min_area_ratio=0.3,
        max_aspect_ratio=10,
        skip_filter=False,
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        ann_file=[
            data_root + 'train.json',
        ],
        img_prefix=[
            data_root + 'train/',
        ],
        filter_empty_gt=False,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'train.json',
            ],
            img_prefix=[
                data_root + 'train/',
            ],
            classes=classes,
            filter_empty_gt=False,
            pipeline=load_pipeline,
        ),
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'valid.json',
        img_prefix=data_root + 'valid/',
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'valid.json',
        img_prefix=data_root + 'valid/',
        classes=classes,
        pipeline=test_pipeline,
    ),
)

custom_hooks = [
    dict(
        type='MultiImageMixAugSwitchHook',
        num_last_epochs=num_last_epochs,
        skip_type_keys=('MosaicWithAffineFlag', 'MixUpWithProb'),
        priority=48,
    ),
]

evaluation = dict(interval=1, metric='bbox')
checkpoint_config = dict(interval=1, max_keep_ckpts=-1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)

auto_scale_lr = dict(enable=False, base_batch_size=16)
