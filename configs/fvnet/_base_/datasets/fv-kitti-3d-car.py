dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Car']
point_cloud_range = [0, -40, -3, 80, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
file_client_args = dict(backend='disk')


fv_size = (620, 190)
img_size = (1242, 375)
train_pipeline = [
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
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ProjectToImage'),
    dict(type='RandomFlipFV', flip_ratio=0.5),
    dict(type='ResizeFV', size=fv_size),
    dict(type='PadFV', size_divisor=32),
    dict(type='DefaultFormatBundleFV', class_names=class_names),
    dict(type='Collect3D', keys=['fv', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1242, 375),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='RandomFlip3D'),
            dict(type='ProjectToImage'),
            dict(type='ResizeFV', size=fv_size),
            dict(type='PadFV', size_divisor=32),
            dict(type='DefaultFormatBundleFV', class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['fv'])
        ]
    )
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            # ann_file=data_root + 'kitti_infos_debug.pkl',
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
        ann_file=data_root + 'kitti_infos_val.pkl',
        # ann_file=data_root + 'kitti_infos_debug.pkl',
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