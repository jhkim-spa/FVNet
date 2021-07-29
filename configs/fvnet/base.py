_base_ = [
    './_base_/models/fvnet_fv.py',
    './_base_/datasets/fv-kitti-3d-car.py',
    './_base_/schedules/step.py',
    '../_base_/default_runtime.py'
]

evaluation = dict(interval=2)
checkpoint_config = dict(interval=2)
