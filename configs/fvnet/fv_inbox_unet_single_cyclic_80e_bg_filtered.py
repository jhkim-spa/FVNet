_base_ = [
    './model/fvnet_fv.py', './dataset/fv-kitti-3d-car_bg_filtered.py',
    '../_base_/schedules/cyclic_40e.py',
    '../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        type='FVNetHeadTest',
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

train_cfg = dict(
    assigner=dict(type='InBoxAssigner'),
    allowed_border=0,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    use_rotate_nms=True,
    nms_across_levels=False,
    nms_thr=0.01,
    score_thr=0.5,
    min_bbox_size=0,
    max_num=50)