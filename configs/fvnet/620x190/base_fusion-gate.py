_base_ = [
    '../model/fvnet_fv.py', '../dataset/fv-kitti-3d-car_620x190_fusion.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    fusion_mode='gate',
    backbone=dict(concat=True, outconv=False),
    bbox_head=dict(feat_channels=131),
    backbone_img=dict(
        type='UNet',
        num_outs=1,
        n_channels=3,
        concat=False,
        outconv=False
    )
)

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

# optimizer
# This schedule is mainly used by models on nuScenes dataset
# lr = 0.003
lr = 0.0015
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

# data = dict(samples_per_gpu=16)
data = dict(samples_per_gpu=8)
