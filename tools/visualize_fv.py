import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import os
import torch
from argparse import ArgumentParser
from copy import deepcopy
from mmcv.parallel import collate, scatter
from PIL import Image
from visualize_utils import (draw_projected_box3d, load_label, load_velo_scan,
                             map_box_to_image, project_cam2_to_velo,
                             project_camera_to_lidar, project_to_image,
                             project_velo_to_cam2, read_calib_file)

from mmdet3d.apis import init_detector
from mmdet3d.apis.open3d_vis import visualize_open3d
from mmdet3d.core.bbox import CameraInstance3DBoxes, get_box_type
from mmdet3d.datasets.pipelines import Compose


def render_image_with_boxes(img, objects, calib, color, classes):
    """
    Show image with 3D boxes
    """
    # projection matrix
    P_rect2cam2 = calib['P2'].reshape((3, 4))

    img1 = np.copy(img)
    for obj in objects:
        if obj.type in classes:
            box3d_pixelcoord = map_box_to_image(obj, P_rect2cam2)
            img1 = draw_projected_box3d(img1, box3d_pixelcoord, color, thickness=1)

    return img1


def render_lidar_with_boxes(pc_velo, objects, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pc_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]
    imgfov_pc_velo = pc_velo[inds, :]

    # Projection matrix
    proj_cam2_2_velo = project_cam2_to_velo(calib)

    # Draw objects on lidar
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        # Project boxes from camera to lidar coordinate
        boxes3d_pts = project_camera_to_lidar(obj.in_camera_coordinate(), proj_cam2_2_velo)



def render_lidar_on_image(pts_velo, img, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    img = deepcopy(img)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pts_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(256 / depth), :]
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                   2, color=tuple(color), thickness=1)
    return img


def get_gt_bboxes(label_path, classes, img_meta):
    rect = img_meta['rect']
    Trv2c = img_meta['Trv2c']
    box_mode_3d = img_meta['box_mode_3d']
    gt_bboxes = []
    with open(label_path, 'r') as f:
        labels = f.readlines()
    for label in labels:
        category = label.split(' ')[0]
        if category in classes:
            loc = label.split(' ')[11: 14]
            dims = label.split(' ')[8: 11]
            rots = [label.split(' ')[14].rstrip()]
            dims = [dims[i] for i in [2, 0, 1]]

            gt_bboxes_3d = np.concatenate([loc, dims, rots]).astype(np.float32)
            gt_bboxes_3d = gt_bboxes_3d.reshape(1, -1)
            gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
                box_mode_3d, np.linalg.inv(rect @ Trv2c))
            gt_bboxes.append(gt_bboxes_3d.tensor)
    if len(gt_bboxes) == 0:
        return None
    gt_bboxes = np.concatenate(gt_bboxes, axis=0)
    return gt_bboxes

def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def get_calib_info(calib_path, extend_matrix=True):
    info = dict()
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                    ]).reshape([3, 4])
    P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                    ]).reshape([3, 4])
    P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                    ]).reshape([3, 4])
    P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                    ]).reshape([3, 4])
    if extend_matrix:
        P0 = _extend_matrix(P0)
        P1 = _extend_matrix(P1)
        P2 = _extend_matrix(P2)
        P3 = _extend_matrix(P3)
    R0_rect = np.array([
        float(info) for info in lines[4].split(' ')[1:10]
    ]).reshape([3, 3])
    if extend_matrix:
        rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        rect_4x4[3, 3] = 1.
        rect_4x4[:3, :3] = R0_rect
    else:
        rect_4x4 = R0_rect

    Tr_velo_to_cam = np.array([
        float(info) for info in lines[5].split(' ')[1:13]
    ]).reshape([3, 4])
    Tr_imu_to_velo = np.array([
        float(info) for info in lines[6].split(' ')[1:13]
    ]).reshape([3, 4])
    if extend_matrix:
        Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
        Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
    info['P0'] = P0
    info['P1'] = P1
    info['P2'] = P2
    info['P3'] = P3
    info['R0_rect'] = rect_4x4
    info['Tr_velo_to_cam'] = Tr_velo_to_cam
    info['Tr_imu_to_velo'] = Tr_imu_to_velo
    return info

def inference_detector(model, pcd, img, calib):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    # get calibration info
    img = Image.open(img)
    width, height = img.size
    info = get_calib_info(calib, extend_matrix=True)
    rect = info['R0_rect'].astype(np.float32)
    Trv2c = info['Tr_velo_to_cam'].astype(np.float32)
    P2 = info['P2'].astype(np.float32)
    lidar2img = P2 @ rect @ Trv2c

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    data = dict(
        pts_filename=pcd,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[],
        img_info={'width': width, 'height': height},
        rect=rect,
        Trv2c=Trv2c,
        lidar2img=lidar2img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data

def _load_points(pts_filename):
    """Private function to load point clouds data.

    Args:
        pts_filename (str): Filename of point clouds data.

    Returns:
        np.ndarray: An array containing point clouds data.
    """
    file_client_args = dict(backend='disk')
    file_client = mmcv.FileClient(**file_client_args)
    try:
        pts_bytes = file_client.get(pts_filename)
        points = np.frombuffer(pts_bytes, dtype=np.float32)
    except ConnectionError:
        mmcv.check_file_exist(pts_filename)
        if pts_filename.endswith('.npy'):
            points = np.load(pts_filename)
        else:
            points = np.fromfile(pts_filename, dtype=np.float32)

    return points

def main():
    parser = ArgumentParser()
    parser.add_argument('--pcd', help='Point cloud folder')
    parser.add_argument('--img', help='Image folder')
    parser.add_argument('--calib', help='Calib folder')
    parser.add_argument('--label', help='Label folder')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.6, help='bbox score threshold')
    args = parser.parse_args()

    model = init_detector(args.config, args.checkpoint, device=args.device)
    pcd_paths = sorted(os.listdir(args.pcd))
    img_paths = sorted(os.listdir(args.img))
    calib_paths = sorted(os.listdir(args.calib))
    label_paths = sorted(os.listdir(args.label))
    for pcd_path, img_path, calib_path, label_path in\
            zip(pcd_paths, img_paths, calib_paths, label_paths):
        idx = pcd_path.split('/')[-1].split('.')[0]
        pcd_path = os.path.join(args.pcd, pcd_path)
        img_path = os.path.join(args.img, img_path)
        calib_path = os.path.join(args.calib, calib_path)
        label_path = os.path.join(args.label, label_path)
        result, data = inference_detector(model, pcd_path, img_path, calib_path)

        calib = read_calib_file(calib_path)
        labels = load_label(label_path)
        pc_velo = load_velo_scan(pcd_path)[:, :3]
        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = rgb.shape
        img1 = render_lidar_on_image(pc_velo, rgb, calib, img_width, img_height)
        img2 = render_image_with_boxes(rgb, labels, calib, color=(255, 0, 0), classes=['Car'])


        points = _load_points(pcd_path).reshape(-1, 4)[:, :3]
        gt_bboxes = get_gt_bboxes(label_path, classes=['Car'],
            img_meta=data['img_metas'][0][0])
        pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
        img = Image.open(img_path)
        img = np.concatenate((img, img1, img2), axis=0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 1000, 800)
        cv2.imshow('Image', img)
        cv2.waitKey(1)

        visualize_open3d(points, pred_bboxes, gt_bboxes, idx=idx)


if __name__ == '__main__':
    main()
