_base_ = [
    'co_dino_5scale_9encoder_lsj_r50_3x_digits_mosaic-mixup.py',
]

sr_data_root = '../nycu-hw2-data_tair_1024/'

data = dict(
    train=dict(
        ann_file=[sr_data_root + 'train.json'],
        img_prefix=[sr_data_root + 'train/'],
        dataset=dict(
            ann_file=[sr_data_root + 'train.json'],
            img_prefix=[sr_data_root + 'train/'],
        ),
    ),
    val=dict(
        ann_file=sr_data_root + 'valid.json',
        img_prefix=sr_data_root + 'valid/',
    ),
    test=dict(
        ann_file=sr_data_root + 'valid.json',
        img_prefix=sr_data_root + 'valid/',
    ),
)
