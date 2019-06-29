from chainer import backend
from chainer_graphics.transform import to_homogenous
import chainer.functions as F

def cam2pixel(K, x):
    """Convert 3D points to pixel coordinates.

    Args:
        K (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 3, 3)`.
            Camera matrices.

        x (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 3, N)`.
            3D points x, y and z.

    Returns:
        ~chainer.Variable:
            A 2-D array of shape `(B, 2, N)`.
            Pixel coordinates u and v.
    """
    B, _, N = x.shape
    x = (x / x[:,2].reshape(-1, 1, N))
    return F.batch_matmul(K, x)[:,:2]

def pixel2cam(K, p, z):
    """Convert pixel coordinates to 3D points

    Args:
        K (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 3, 3)`.
            Camera matrices.

        p (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 2, N)`.
            Pixel coordinates u and v.

        z (:class `~chainer.Variable` or :ref:`ndarray`):
            A 1-D array of length `(B, N)`.
            z-coordinates of corresponding (u, v).

    Returns:
        ~chainer.Variable:
            A 2-D array of shape `(B, 3, N)`.
            3D points x, y and z.
    """
    B, _, N = p.shape
    z = z.reshape(-1, 1, N)
    return z * (F.batch_inv(K) @ to_homogenous(p))
