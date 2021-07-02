model = dict(
    type='FVNet',
    depth_range=(0, 80),
    backbone=dict(
        type='UNet',
        num_outs=1,
        n_channels=5),
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

find_unused_parameters = True