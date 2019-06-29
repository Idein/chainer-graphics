from chainer import backend
import chainer.functions as F

def camera_matrix(focal, offset):
    """Create camera matrix from focal lengths and principal point offsets

    Args:
        focal (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 2)`

        offset (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 2)`

    Returns:
        ~chainer.Variable:
            Camera matrices.
            A 3-D array of shape `(B, 3, 3)`
    """
    xp = backend.get_array_module(focal)

    fx = focal[:, 0]
    fy = focal[:, 1]
    cx = offset[:, 0]
    cy = offset[:, 1]

    zeros = xp.zeros_like(fx)
    ones  = xp.ones_like(fx)

    K = F.stack([
        fx, zeros, cx,
        zeros, fy, cy,
        zeros, zeros, ones], axis=1).reshape(-1, 3, 3)
    return K

