optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-07, by_epoch=True)
total_epochs = 1200
checkpoint_config = dict(interval=10)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoints/textdet/dbnet/res50dcnv2_synthtext.pth'
resume_from = 'dbnet/latest.pth'
workflow = [('train', 1)]
model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPNC', in_channels=[256, 512, 1024, 2048], lateral_channels=256),
    bbox_head=dict(
        type='DBHead',
        text_repr_type='quad',
        in_channels=256,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=True)),
    train_cfg=None,
    test_cfg=None)
dataset_type = 'IcdarDataset'
data_root = 'data/mergeCleanVBS_VIN_Tempo'
img_norm_cfg = dict(
    mean=[122.67891434, 116.66876762, 104.00698793],
    std=[255, 255, 255],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=0.12549019607843137, saturation=0.5),
    dict(
        type='Normalize',
        mean=[122.67891434, 116.66876762, 104.00698793],
        std=[255, 255, 255],
        to_rgb=False),
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.5], {
            'cls': 'Affine',
            'rotate': [-10, 10]
        }, ['Resize', [0.5, 3.0]]]),
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.4),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(4068, 1024), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[255, 255, 255],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='IcdarDataset',
        ann_file='data/mergeCleanVBS_VIN_Tempo/instances_train.json',
        img_prefix='data/mergeCleanVBS_VIN_Tempo/imgs',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='LoadTextAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='ColorJitter',
                brightness=0.12549019607843137,
                saturation=0.5),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[255, 255, 255],
                to_rgb=False),
            dict(
                type='ImgAug',
                args=[['Fliplr', 0.5], {
                    'cls': 'Affine',
                    'rotate': [-10, 10]
                }, ['Resize', [0.5, 3.0]]]),
            dict(type='EastRandomCrop', target_size=(640, 640)),
            dict(type='DBNetTargets', shrink_ratio=0.4),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr',
                    'gt_thr_mask'
                ])
        ]),
    val=dict(
        type='IcdarDataset',
        ann_file='data/mergeCleanVBS_VIN_Tempo/instances_test.json',
        img_prefix='data/mergeCleanVBS_VIN_Tempo/imgs',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(4068, 1024),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(4068, 1024),
                        keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[122.67891434, 116.66876762, 104.00698793],
                        std=[255, 255, 255],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='IcdarDataset',
        ann_file='data/mergeCleanVBS_VIN_Tempo/instances_test.json',
        img_prefix='data/mergeCleanVBS_VIN_Tempo/imgs',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(4068, 1024),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(4068, 1024),
                        keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[122.67891434, 116.66876762, 104.00698793],
                        std=[255, 255, 255],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=100, metric='hmean-iou')
work_dir = 'dbnet/'
gpu_ids = range(0, 1)
