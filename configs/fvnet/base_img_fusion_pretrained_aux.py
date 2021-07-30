_base_ = [
    './_base_/models/fvnet_fv.py',
    './_base_/datasets/fv-kitti-3d-car-fusion.py',
    './_base_/schedules/step.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone_img=dict(
        type='UnetResNet',
        input_channels=3,
        num_classes=1,
        pretrained_seg='pretrained/best_model.pth',
    ),
    bbox_head=dict(
        in_channels=99,
        feat_channels=99
    ),
    aux_head=dict(
        type='FVNetAuxHead',
        num_classes=1,
        in_channels=32,
        bbox_coder=dict(type='DeltaXYWHBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0)
    )
)

evaluation = dict(interval=2)
checkpoint_config = dict(interval=2)

find_unused_parameters = True