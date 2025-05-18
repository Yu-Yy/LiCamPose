# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import division
import numpy as np


def unfold_camera_param(camera):
    R = camera['R']
    T = camera['T']
    fx = camera['fx']
    fy = camera['fy']
    # f = 0.5 * (camera['fx'] + camera['fy'])
    f = np.array([[fx], [fy]]).reshape(-1, 1)
    c = np.array([[camera['cx']], [camera['cy']]]).reshape(-1, 1)
    k = camera['k']
    p = camera['p']
    return R, T, f, c, k, p


def project_point_radial(x, R, T, f, c, k, p):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    n = x.shape[0]
    # xcam = R.dot(x.T - T)  # cam parameter bbug
    xcam = R @ x.T + T
    y = xcam[:2] / (xcam[2]+1e-5)
    # print(xcam[2])

    r2 = np.sum(y**2, axis=0)
    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, n)),
                           np.array([r2, r2**2, r2**3]))
    tan = p[0] * y[1] + p[1] * y[0]
    y = y * np.tile(radial + 2 * tan,
                    (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
    ypixel = np.multiply(f, y) + c
    return ypixel.T


def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera)
    return project_point_radial(x, R, T, f, c, k, p)

def project_pose_sync(x, projectionM, width = 2048, height = 1536):
    '''
    Args:
        x is the N x 3 world coordinates
        projectionM is the Projection matrix
    '''
    cat_coord = np.concatenate([x, np.ones((x.shape[0],1))], axis=-1)
    ndc_coord = cat_coord @ projectionM.T
    screen_coord = np.zeros((x.shape[0], 2))
    screen_coord[:,0] = (ndc_coord[:,0] / ndc_coord[:,-1] + 1) * 0.5 * width
    screen_coord[:,1] = height * (1 - (ndc_coord[:,1] / ndc_coord[:,-1] + 1) * 0.5)
    return screen_coord


def undistort_pose2d(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera)
    return undistort_point2d(x, f, c, k, p)

def undistort_point2d(x, f, c, k, p):
    """
    Args
        x: Nx2 points in pixel space
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    n = x.shape[0]

    xcam = (x.T - c) / f  # [2, N]

    # === remove camera distortion (approx)
    r = xcam[0, :] * xcam[0, :] + xcam[1, :] * xcam[1, :]
    d = 1 - k[0] * r - k[1] * r * r - k[2] * r * r * r
    u = xcam[0, :] * d - 2 * p[0] * xcam[0, :] * xcam[1, :] - p[1] * (r + 2 * xcam[0, :] * xcam[0, :])
    v = xcam[1, :] * d - 2 * p[1] * xcam[0, :] * xcam[1, :] - p[0] * (r + 2 * xcam[1, :] * xcam[1, :])
    xcam[0, :] = u
    xcam[1, :] = v

    ypixel = np.multiply(f, xcam) + c

    return ypixel.T

