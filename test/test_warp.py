import numpy as np
import cv2

import chainer_graphics.image as I

eps = 1e-5

def cosine_similarity(image0, image1):
    v0 = image0.flatten()
    v1 = image1.flatten()
    return v0.dot(v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))

def test_pixelcoords():
    image = np.random.randn(2, 3, 100, 200)
    ps = I.pixel_coords(np, 100, 200).data
    us, vs = np.meshgrid(np.arange(200), np.arange(100))
    assert(np.all(ps[0] == us))
    assert(np.all(ps[1] == vs))

def test_warp_affine():
    identity = np.array([[1, 0, 0], [0, 1, 0]])
    mat = identity + np.random.randn(2, 3)/10

    image = cv2.imread("test/lena.png").astype('f')
    H, W, _ = image.shape

    ref_image = cv2.warpAffine(image, mat, (W, H))

    image = np.expand_dims(image.transpose((2, 0, 1)), axis=0)
    mat = np.expand_dims(mat, axis=0)
    warped_image = I.warp_affine(image.astype(float), mat).data
    warped_image = warped_image.transpose((0, 2, 3, 1)).reshape((H, W, 3))

    assert(1 - cosine_similarity(ref_image, warped_image) < eps)
