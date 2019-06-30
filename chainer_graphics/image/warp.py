from chainer import backend
import chainer.functions as F
from chainer_graphics.image import *

def affine(A, t, x):
    """Compute Ax+t

    Args:
        A (:class `~chainer.Variable` or :ref:`ndarray`):
            A 3-D array of shape `(B, M, M)`
        t (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, M)`
        x (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, M, N)`

    Returns:
        ~chainer.Variable:
            A 2-D array of shape `(B, M, N)`
    """
    return A @ x + F.expand_dims(t, axis=2)

def inverse_affine(A, t, x):
    """Compute A^-1(x-t)

    Args:
        A (:class `~chainer.Variable` or :ref:`ndarray`):
            A 3-D array of shape `(B, M, M)`
        t (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, M)`
        x (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, M, N)`

    Returns:
        ~chainer.Variable:
            A 2-D array of shape `(B, M, N)`
    """

    return F.batch_inv(A) @ (x - F.expand_dims(t, axis=2))

def warp_dense(image, ps):
    """Dense image warping

    Args:
        image (:class `~chainer.Variable` or :ref:`ndarray`):
            A 4-D array of shape `(B, C, H, W)`.

        ps (:class `~chainer.Variable` or :ref:`ndarray`):
            A 4-D array of shape `(B, 2, H, W)`
            Pixel coordinates in source images.

    Returns:
        ~chainer.Variable:
            Warped image.
            A 4-D array of shape `(B, C, H, W)`.
    """
    xp = backend.get_array_module(image)
    B, _, H, W = image.shape
    ps = 2 * ps / xp.array([W-1, H-1]).reshape(-1, 2, 1, 1) - 1
    ps = ps.reshape(B, 2, H, W)

    return F.spatial_transformer_sampler(image, ps)

def warp_affine(image, mat):
    """Warp images with affine transformation

    Args:
        image (:class `~chainer.Variable` or :ref:`ndarray`):
            A 4-D array of shape `(B, C, H, W)`.

        mat (:class `~chainer.Variable` or :ref:`ndarray`):
            Affine transformation matrices [[a, b, tx], [c, d, ty]].
            A 3-D array of shape `(B, 2, 3)`.

    Returns:
        ~chainer.Variable:
            Warped image.
            A 4-D array of shape `(B, C, H, W)`.
    """
    xp = backend.get_array_module(image)
    B, _, H, W = image.shape

    ps1 = pixel_coords(xp, H, W, mat.dtype).reshape(1, 2, -1)
    ps0 = inverse_affine(mat[:, :, :2], mat[:, :, 2], ps1)
    return warp_dense(image, ps0.reshape(1, 2, H, W))

def warp_perspective(image, mat):
    """Warp images with perspective transformation

    Args:
        image (:class `~chainer.Variable` or :ref:`ndarray`):
            A 4-D array of shape `(B, C, H, W)`.

        mat (:class `~chainer.Variable` or :ref:`ndarray`):
            Perspective transformaion matrices.
            A 3-D array of shape `(B, 3, 3)`.

    Returns:
        ~chainer.Variable:
            Warped image.
            A 4-D array of shape `(B, C, H, W)`.
    """
    xp = backend.get_array_module(image)
    B, _, H, W = image.shape

    ps1 = pixel_coords(xp, H, W, mat.dtype).reshape(1, 2, -1)
    num   = affine(mat[:,:2,:2], mat[:,:2,2], ps1)
    denom = affine(mat[:,2,:2].reshape(-1, 1, 2), mat[:,2,2].reshape(-1, 1), ps1)
    ps0 = num / denom
    return warp_dense(image, ps0.reshape(1, 2, H, W))
