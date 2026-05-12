"""HTC Swin-B Copy-Paste with mild photometric augmentation ablation."""

_base_ = './htc_swin_b_copypaste.py'

backend_args = None

copy_paste_load_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=24,
        contrast_range=(0.75, 1.25),
        saturation_range=(0.75, 1.25),
        hue_delta=8),
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

train_dataloader = dict(
    dataset=dict(
        dataset=dict(pipeline=copy_paste_load_pipeline)))

test_evaluator = dict(
    outfile_prefix=(
        './work_dirs/'
        'htc_swin_b_copypaste_photometric/test'))

work_dir = (
    './work_dirs/'
    'htc_swin_b_copypaste_photometric')
