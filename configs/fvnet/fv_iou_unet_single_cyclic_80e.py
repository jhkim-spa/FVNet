_base_ = [
    './model/fvnet_fv.py', './dataset/fv-kitti-3d-car.py',
    '../_base_/schedules/cyclic_40e.py',
    '../_base_/default_runtime.py'
]

train_cfg = dict(
    assigner=dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
    allowed_border=0,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    use_rotate_nms=True,
    nms_across_levels=False,
    nms_thr=0.01,
    score_thr=0.3,
    min_bbox_size=0,
    max_num=50)