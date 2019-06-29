from chainer import backend
import chainer.functions as F

def euler2rot(angles):
    """Create rotation matrices from euler angles

    Args:
        angles (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 3)`
            Euler angles along with x, y and z axis respectively.

    Returns:
        ~chainer.Variable:
            Rotation matrices.
            A 3-D array of shape `(B, 3, 3)`
    """

    xp = backend.get_array_module(angles)

    xs = angles[:, 0]
    ys = angles[:, 1]
    zs = angles[:, 2]

    zeros = xp.zeros_like(xs)
    ones  = xp.ones_like(xs)

    cosxs = F.cos(xs)
    sinxs = F.sin(xs)
    Rx = F.stack([ ones, zeros,  zeros,
                  zeros, cosxs, -sinxs,
                  zeros, sinxs,  cosxs], axis=1).reshape(-1, 3, 3)

    cosys = F.cos(ys)
    sinys = F.sin(ys)
    Ry = F.stack([ cosys, zeros, sinys,
                   zeros,  ones, zeros,
                  -sinys, zeros, cosys], axis=1).reshape(-1, 3, 3)

    coszs = F.cos(zs)
    sinzs = F.sin(zs)
    Rz = F.stack([coszs, -sinzs, zeros,
                  sinzs,  coszs, zeros,
                  zeros,  zeros,  ones], axis=1).reshape(-1, 3, 3)

    return Rz @ Ry @ Rx


def rot2euler(matrices):
    """Calculate euler angles from rotation matrices

    Args:
        matrices (:class `~chainer.Variable` or :ref:`ndarray`):
            Rotation matrices. A 3-D array of shape `(B, 3, 3)`

    Returns:
        ~chainer.Variable:
            A 2-D array of shape `(B, 3)`
            Euler angles along with x, y, and z axis respectively.
    """

    xs = F.arctan2(matrices[:,2,1], matrices[:,2,2])
    ys = F.arcsin(-matrices[:,2,0])
    zs = F.arctan2(matrices[:,1,0], matrices[:,0,0])

    return F.stack([xs, ys, zs], axis=1)
