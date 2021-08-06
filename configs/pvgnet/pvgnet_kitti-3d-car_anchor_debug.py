_base_ = ['./pvgnet_kitti-3d-car_anchor.py']

data = dict(
    train=dict(dataset=dict(ann_file='data/kitti/kitti_infos_debug.pkl')),
    val=dict(ann_file='data/kitti/kitti_infos_debug.pkl'),
    test=dict(ann_file='data/kitti/kitti_infos_debug.pkl')
)