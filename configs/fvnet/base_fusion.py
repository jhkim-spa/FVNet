_base_ = [
    './_base_/models/fvnet_fv.py',
    './_base_/datasets/fv-kitti-3d-car-fusion.py',
    './_base_/schedules/step.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone_img=dict(
        type='UNet',
        num_outs=1,
        n_channels=3,
        concat=False),
)

evaluation = dict(interval=2)
checkpoint_config = dict(interval=2)
