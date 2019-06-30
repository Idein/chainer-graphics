import numpy as np
import cv2

import chainer_graphics.camera as C

BATCH = 2
POINTS = 10

def random_array(*args):
    return np.random.randn(*args).astype('float32')

def test_cam2pixel():
    focal = 1 + random_array(1, 2) / 100
    offset = random_array(1, 2) / 100
    K = C.camera_matrix(focal, offset)
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    dist = random_array(1, 8) * 1e-5

    x0 = random_array(1, POINTS, 3)

    p0, _ = cv2.projectPoints(x0[0], rvec, tvec, K.data[0], dist)
    p1 = C.cam2pixel(K, x0.transpose((0, 2, 1)), dist).data.transpose((2, 0, 1))
    assert(np.allclose(p0, p1, rtol=1e-4))

def test_pixel2cam():
    focal = 1 + random_array(1, 2) / 100
    offset = random_array(1, 2) / 100
    dist = random_array(1, 8) * 1e-5

    p = random_array(1, POINTS, 2)
    z = random_array(1, POINTS, 1)
    K = C.camera_matrix(focal, offset)

    q = cv2.undistortPoints(p, K[0].data, dist[0])
    x = z * q[:, :, 0:1]
    y = z * q[:, :, 1:2]
    x0 = np.concatenate((x, y, z), axis=2)

    x1 = C.pixel2cam(K, p.transpose((0,2,1)), z[:,:,0], dist).data.transpose((0,2,1))

    assert(np.allclose(x0, x1))

def test_reversibility():
    x = 2*np.random.random((BATCH, 2, POINTS)).astype(np.float32) - 1
    z = np.ones((BATCH, 1, POINTS), np.float32)
    x0 = np.concatenate((x, z), axis=1)
    focal = 1 + random_array(BATCH, 2) / 100
    offset = random_array(BATCH, 2) / 100
    dist = random_array(1, 8) * 1e-5

    K = C.camera_matrix(focal, offset)
    p = C.cam2pixel(K, x0, dist)
    x1 = C.pixel2cam(K, p, z, dist).data
    assert(np.allclose(x0, x1))
