"""HTC Swin-B FlipD4 pseudo-label self-finetuning for 8 epochs."""

_base_ = './htc_swin_b_copypaste_all_random-erasing_flipd4.py'

backend_args = None
pseudo_ann = (
    './data/processed/annotations/'
    'instances_train_all_plus_pseudo_ens37_34_thr0p20.json')

copy_paste_load_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,
         with_seg=False),
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

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            data_root='./',
            ann_file=pseudo_ann,
            data_prefix=dict(_delete_=True, img=''),
            pipeline=copy_paste_load_pipeline)))

max_epochs = 8
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs,
                 val_interval=999999)
val_cfg = None
val_dataloader = None
val_evaluator = None

optim_wrapper = dict(
    optimizer=dict(lr=2e-5))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[6],
        gamma=0.2)
]

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=-1))

load_from = (
    './work_dirs/'
    'htc_swin_b_copypaste_all_random-erasing_flipd4/epoch_24.pth')
resume = False

test_evaluator = dict(
    outfile_prefix=(
        './work_dirs/'
        'htc_swin_b_copypaste_all_random-erasing_flipd4_pseudolabel_ft/test'))

work_dir = (
    './work_dirs/'
    'htc_swin_b_copypaste_all_random-erasing_flipd4_pseudolabel_ft')
