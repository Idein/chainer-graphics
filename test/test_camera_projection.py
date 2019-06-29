import numpy as np
import cv2

import chainer_graphics.camera as C

BATCH = 10
POINTS = 100

def test_projection():
    x0 = np.random.randn(BATCH, 3, POINTS).astype('float32')
    focal = 1 + np.random.randn(BATCH, 2).astype('float32') / 100
    offset = np.random.randn(BATCH, 2).astype('float32') / 100

    K = C.camera_matrix(focal, offset)
    p = C.cam2pixel(K, x0)
    x1 = C.pixel2cam(K, p, x0[:,2]).data
    assert(np.allclose(x0, x1))
