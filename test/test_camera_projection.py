import numpy as np
import cv2

import chainer_graphics.camera as C

BATCH = 2
POINTS = 3

def random_array(*args):
    return np.random.randn(*args).astype('float32')

def test_cam2pixel():
    focal = 1 + random_array(1, 2) / 100
    offset = random_array(1, 2) / 100
    K = C.camera_matrix(focal, offset)
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    dist = random_array(1, 8) / 100

    x0 = random_array(1, POINTS, 3)

    p0, _ = cv2.projectPoints(x0[0], rvec, tvec, K.data[0], dist)
    p1 = C.cam2pixel(K, x0.transpose((0, 2, 1)), dist).data.transpose((2, 0, 1))
    assert(np.allclose(p0, p1, rtol=1e-4))

def test_pixel2cam():
    dist = random_array(1, 8) / 1000

    x0 = random_array(1, POINTS, 2)

    p0 = cv2.undistortPoints(x0, np.eye(3), dist)
    p1 = C.undistort(dist, x0.transpose((0, 2, 1))).data.transpose((0, 2, 1))
    assert(np.allclose(p0, p1, rtol=1e-4))

def test_reversibility():
    x0 = np.random.randn(BATCH, 3, POINTS).astype('float32')
    focal = 1 + np.random.randn(BATCH, 2).astype('float32') / 100
    offset = np.random.randn(BATCH, 2).astype('float32') / 100

    K = C.camera_matrix(focal, offset)
    p = C.cam2pixel(K, x0)
    x1 = C.pixel2cam(K, p, x0[:,2]).data
    assert(np.allclose(x0, x1))
