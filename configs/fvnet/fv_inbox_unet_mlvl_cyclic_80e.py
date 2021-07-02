_base_ = [
    './fv_inbox_unet_single_cyclic_80e.py'
]

model = dict(
    depth_range=(0, 20, 40, 60, 80),
    backbone=dict(
        num_outs=4
    )
)