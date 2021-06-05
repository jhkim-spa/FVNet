import numpy as np


def render_lidar_on_image(pts_velo, calib, img_width, img_height):
    reflectance = pts_velo[:, -1]
    pts_velo = pts_velo[:, :3]
    # projection matrix (project from velo2cam2)
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
    # imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    # imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose() ## ??????????????????????????????
    
    # Reflectance
    reflectance = reflectance[inds]

    pc_projected = np.zeros((img_height, img_width, 5), dtype=np.float32)
    x_coords = np.trunc(imgfov_pc_pixel[0]).astype(np.int)
    y_coords = np.trunc(imgfov_pc_pixel[1]).astype(np.int)
    pc_projected[y_coords, x_coords, :3] = imgfov_pc_velo
    pc_projected[y_coords, x_coords, 3] = reflectance
    flag_channel = (pc_projected[:, :, 0] != 0)
    pc_projected[:, :, -1] = flag_channel

    return pc_projected


def project_velo_to_cam2(calib):
    P_velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    R_ref2rect = np.eye(4)
    R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
    return proj_mat


def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]


def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data
