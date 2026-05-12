"""HTC Swin-B with flip-D4, RandomErasing, and rare-class Copy-Paste."""

_base_ = './htc_swin_b_copypaste_all_random-erasing_flipd4.py'

copy_paste_pipeline = [
    dict(
        type='RareClassCopyPaste',
        rare_classes=(2, 3),
        rare_image_prob=0.80,
        rare_class_weight=10.0,
        min_num_pasted=20,
        min_rare_pasted=8,
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
        pipeline=copy_paste_pipeline))

test_evaluator = dict(
    outfile_prefix=(
        './work_dirs/'
        'htc_swin_b_copypaste_all_random-erasing_flipd4_rarecp/test'))

work_dir = (
    './work_dirs/'
    'htc_swin_b_copypaste_all_random-erasing_flipd4_rarecp')
