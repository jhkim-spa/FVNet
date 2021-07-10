_base_ = [
    '../dataset/fv-kitti-3d-car_anchor.py',
    '../../_base_/default_runtime.py'
]

model=dict(
    type='FVNet',
    use_anchor=True,
    backbone=dict(
        type='ResNet',
        in_channels=5,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    bbox_head=dict(
        type='FVNetAnchorHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='LineAnchorGenerator',
            # dist_list=list(range(0, 80, 2)),
            # num_bins=8,
            dist_list=list(range(0, 80, 1)),
            num_bins=16,
            ranges=[[0, -40.0, -1.78, 70.4, 40.0, -1.78]],
            sizes=[[1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True,
            scales=[1, 1, 1, 1, 1]),
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

train_cfg = dict(
    assigner=dict(  # for Car
        type='MaxIoUAssigner',
        # iou_calculator=dict(type='BboxOverlapsNearest3D'),
        iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
        pos_iou_thr=0.6,
        neg_iou_thr=0.45,
        min_pos_iou=0.45,
        ignore_iof_thr=-1),
    allowed_border=0,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    use_rotate_nms=True,
    nms_across_levels=False,
    nms_thr=0.01,
    score_thr=0.5,
    min_bbox_size=0,
    nms_pre=100,
    max_num=50)

# optimizer
# This schedule is mainly used by models on nuScenes dataset
lr = 0.003
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup=None,
    step=[134, 183])
momentum_config = None
# runtime settings
total_epochs = 200

data = dict(samples_per_gpu=16)
