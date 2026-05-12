"""HTC Swin-B Copy-Paste with denser inference for crowded cell images."""

_base_ = './htc_swin_b_copypaste.py'

model = dict(
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=6000),
        rcnn=dict(
            score_thr=0.001,
            max_per_img=1000)))

test_evaluator = dict(
    outfile_prefix=(
        './work_dirs/'
        'htc_swin_b_copypaste_dense_infer/test'))

work_dir = (
    './work_dirs/'
    'htc_swin_b_copypaste_dense_infer')
