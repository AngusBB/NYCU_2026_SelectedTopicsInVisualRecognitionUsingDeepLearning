"""Fine-tune HTC Swin-B with higher-resolution mask supervision."""

_base_ = './htc_swin_b_copypaste_random-erasing.py'

data_root = './data/processed/'

model = dict(
    roi_head=dict(
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=28, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=False,
                roi_feat_size=28,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=4,
                loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_mask=True,
                    loss_weight=2.0)),
            dict(
                type='HTCMaskHead',
                roi_feat_size=28,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=4,
                loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_mask=True,
                    loss_weight=2.0)),
            dict(
                type='HTCMaskHead',
                roi_feat_size=28,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=4,
                loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_mask=True,
                    loss_weight=2.0))
        ]),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=56,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=56,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=56,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=6000),
        rcnn=dict(
            score_thr=0.001,
            max_per_img=2200,
            mask_thr_binary=0.60)))

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            ann_file='annotations/instances_train_all.json',
            data_prefix=dict(
                img='images/train_all/',
                seg='semantic/train_all/'))))

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
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=-1,
        save_best=None))

load_from = (
    './work_dirs/'
    'htc_swin_b_copypaste_all_random-erasing/epoch_27.pth')
resume = False

test_evaluator = dict(
    outfile_prefix=(
        './work_dirs/'
        'htc_swin_b_copypaste_all_random-erasing_mask56_ft/test'))

work_dir = (
    './work_dirs/'
    'htc_swin_b_copypaste_all_random-erasing_mask56_ft')
