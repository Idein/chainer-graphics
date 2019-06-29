import numpy as np
import cv2

import chainer_graphics.camera as C

BATCH = 10
POINTS = 100

def test_cam2pixel():
    x0 = np.random.randn(1, POINTS, 3).astype('float32')
    focal = 1 + np.random.randn(1,2).astype('float32') / 100
    offset = np.random.randn(1,2).astype('float32') / 100
    K = C.camera_matrix(focal, offset)

    rvec = np.zeros(3)
    tvec = np.zeros(3)

    p0, _ = cv2.projectPoints(x0[0], rvec, tvec, K.data[0], None)
    p1 = C.cam2pixel(K, x0.transpose((0, 2, 1))).data.transpose((2, 0, 1))
    assert(np.allclose(p0, p1))

def test_reversibility():
    x0 = np.random.randn(BATCH, 3, POINTS).astype('float32')
    focal = 1 + np.random.randn(BATCH, 2).astype('float32') / 100
    offset = np.random.randn(BATCH, 2).astype('float32') / 100

    K = C.camera_matrix(focal, offset)
    p = C.cam2pixel(K, x0)
    x1 = C.pixel2cam(K, p, x0[:,2]).data
    assert(np.allclose(x0, x1))
