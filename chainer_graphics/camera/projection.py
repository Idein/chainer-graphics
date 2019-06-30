from chainer import backend
import chainer.functions as F

from chainer_graphics.transform import to_homogenous
from .distortion import distort, undistort

def cam2pixel(K, x, D=None):
    """Convert 3D points to pixel coordinates.

    Args:
        K (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 3, 3)`.
            Camera matrices.

        D (:class `~chainer.Variable` or :ref:`ndarray`):
            Distortion coefficients.
            A 2-D array of shape `(B, K)`
            K is 4 or 5 or 8. The elements corresponds to
            (k1, k2, p1, p2, [k3, [k4, k5 k6]])
            respectively.


        x (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 3, N)`.
            3D points x, y and z.

    Returns:
        ~chainer.Variable:
            A 2-D array of shape `(B, 2, N)`.
            Pixel coordinates u and v.
    """
    B, _, N = x.shape
    x = x / x[:, 2, None, :]
    if D is not None:
        x = to_homogenous(distort(D, x[:,:2,:]))
    x = F.batch_matmul(K, x)[:,:2]
    print(x.shape)
    return x

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
