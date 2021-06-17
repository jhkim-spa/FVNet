_base_ = [
    './fv_inbox_unet_single_cyclic_80e.py'
]

model = dict(
    depth_wise=False,
    backbone=dict(
        num_outs=4
    )
)