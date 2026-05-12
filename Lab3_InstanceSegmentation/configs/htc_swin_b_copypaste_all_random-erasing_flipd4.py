"""HTC Swin-B Copy-Paste + RandomErasing with built-in D4 flip subgroup."""

_base_ = './htc_swin_b_copypaste_random-erasing.py'

backend_args = None

copy_paste_load_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,
         with_seg=True),
    dict(
        type='RandomFlip',
        prob=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal']),
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
    dict(
        type='RandomErasing',
        n_patches=(1, 3),
        ratio=(0.03, 0.12),
        squared=False,
        bbox_erased_thr=0.85,
        img_border_value=(128, 128, 128),
        mask_border_value=0,
        seg_ignore_label=255),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            ann_file='annotations/instances_train_all.json',
            data_prefix=dict(
                img='images/train_all/',
                seg='semantic/train_all/'),
            pipeline=copy_paste_load_pipeline),
        pipeline=copy_paste_pipeline))

max_epochs = 36
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs,
                 val_interval=999999)
val_cfg = None
val_dataloader = None
val_evaluator = None

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=-1))

test_evaluator = dict(
    outfile_prefix=(
        './work_dirs/'
        'htc_swin_b_copypaste_all_random-erasing_flipd4/test'))

work_dir = (
    './work_dirs/'
    'htc_swin_b_copypaste_all_random-erasing_flipd4')
