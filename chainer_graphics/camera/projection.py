from chainer import backend
import chainer.functions as F

from chainer_graphics.transform import to_homogenous
from .distortion import distort_points, undistort_points

def cam2pixel(K, x, D=None):
    """Convert 3D points to pixel coordinates.

    Args:
        K (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 3, 3)`.
            Camera matrices.

        x (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 3, N)`.
            3D points x, y and z.

        D (:class `~chainer.Variable` or :ref:`ndarray`):
            Distortion coefficients.
            A 2-D array of shape `(B, K)`
            K is 4 or 5 or 8. The elements corresponds to
            (k1, k2, p1, p2, [k3, [k4, k5 k6]])
            respectively.

    Returns:
        ~chainer.Variable:
            A 2-D array of shape `(B, 2, N)`.
            Pixel coordinates u and v.
    """
    B, _, N = x.shape
    x = x / x[:, 2, None, :]
    if D is not None:
        x = to_homogenous(distort_points(D, x[:,:2,:]))
    return F.batch_matmul(K, x)[:,:2]

def pixel2cam(K, p, z, D=None):
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

        D (:class `~chainer.Variable` or :ref:`ndarray`):
            Distortion coefficients.
            A 2-D array of shape `(B, K)`
            K is 4 or 5 or 8. The elements corresponds to
            (k1, k2, p1, p2, [k3, [k4, k5 k6]])
            respectively.

    Returns:
        ~chainer.Variable:
            A 2-D array of shape `(B, 3, N)`.
            3D points x, y and z.
    """
    B, _, N = p.shape
    z = z[:, None, :]
    q = F.batch_inv(K) @ to_homogenous(p)
    if D is not None:
        q = undistort_points(D, q[:, :2, :])
        q = to_homogenous(q)
    return z * q
