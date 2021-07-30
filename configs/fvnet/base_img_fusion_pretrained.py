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
    )
)

evaluation = dict(interval=2)
checkpoint_config = dict(interval=2)
