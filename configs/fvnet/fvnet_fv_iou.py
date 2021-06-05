# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
file_client_args = dict(backend='disk')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_scale = (1242, 375)
img_scale = (1696, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomFlip3D', sync_2d=True),
    dict(type='ProjectToImage'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='ResizeFV'),
    dict(type='Pad', size_divisor=32),
    dict(type='PadFV', size_divisor=32),
    dict(type='DefaultFormatBundleFV', class_names=class_names),
    dict(type='Collect3D', keys=['fv', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='RandomFlip3D', sync_2d=True),
            dict(type='ProjectToImage'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Resize', img_scale=img_scale, keep_ratio=True),
            dict(type='ResizeFV'),
            dict(type='Pad', size_divisor=32),
            dict(type='PadFV', size_divisor=32),
            dict(type='DefaultFormatBundleFV', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['fv'])
        ]
    )
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=200,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            # ann_file=data_root + 'kitti_infos_train.pkl',
            ann_file=data_root + 'kitti_infos_debug.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file=data_root + 'kitti_infos_val.pkl',
        ann_file=data_root + 'kitti_infos_debug.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        # ann_file=data_root + 'kitti_infos_debug.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

evaluation = dict(interval=1)

model = dict(
    type='FVNet',
    backbone_fv=dict(
        type='UNet',
        n_channels=5,
        n_classes=1),
    backbone_img=dict(
        type='UNet',
        n_channels=3,
        n_classes=1),
    bbox_head=dict(
        type='FVNetHead',
        anchor_cfg =dict(size=[1.6, 3.9, 1.56],
                         rotation=[0, 1.57]),
        num_classes=1,
        in_channels=64,
        feat_channels=64,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)))
# model training and testing settings
train_cfg = dict(
    assigner=dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
    # assigner=dict(type='InBoxAssigner'),
    allowed_border=0,
    pos_weight=-1,
    debug=False)
# train_cfg = dict(
#     assigner=dict(type='InBoxAssigner'),
#     allowed_border=0,
#     pos_weight=-1,
#     debug=False)
test_cfg = dict(
    use_rotate_nms=True,
    nms_across_levels=False,
    nms_thr=0.01,
    score_thr=0.3,
    min_bbox_size=0,
    nms_pre=500,
    max_num=50)

lr = 0.0018
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)
total_epochs = 40

checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = True
