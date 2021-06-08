_base_ = [
    './model/fvnet_fv.py', './dataset/fv-kitti-3d-car.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

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