import numpy as np
import cv2
import chainer_graphics.camera as C
from util import cosine_similarity, eps

def test_undistort_image():
    image = cv2.imread("test/lena.png").astype(np.float32)
    H, W, _ = image.shape
    K = np.array([[
        [1.0, 0.0, W/2],
        [0.0, 1.0, H/2],
        [0.0, 0.0, 1.0]
        ]], np.float32)
    dist = np.array([[1e-6, 0.0, 1e-5, 0.0]], np.float32)

    ref_image = cv2.undistort(image, K[0], dist[0])

    image = np.expand_dims(image.transpose((2, 0, 1)), axis=0)
    warped_image = C.undistort_image(K, dist, image)[0].data
    warped_image = warped_image.transpose((1, 2, 0))

    #cv2.imwrite('ref.png', ref_image)
    #cv2.imwrite('warped.png', warped_image)

    assert(1 - cosine_similarity(ref_image, warped_image) < eps)

def test_reversibility():
    image = cv2.imread("test/lena.png").astype(np.float32)
    H, W, _ = image.shape
    K = np.array([[
        [1.0, 0.0, W/2],
        [0.0, 1.0, H/2],
        [0.0, 0.0, 1.0]
        ]], np.float32)
    dist = np.array([[1e-6, 0.0, 1e-5, 0.0]], np.float32)

    image0 = np.expand_dims(image.transpose((2, 0, 1)), axis=0)
    distorted_image = C.undistort_image(K, dist, image0).data
    image1 = C.distort_image(K, dist, distorted_image).data

    #cv2.imwrite('distorted.png', distorted_image[0].transpose((1, 2, 0)))
    #cv2.imwrite('image0.png', image0[0].transpose((1, 2, 0)))
    #cv2.imwrite('image1.png', image1[0].transpose((1, 2, 0)))

    assert(1 - cosine_similarity(image0, image1) < eps)
