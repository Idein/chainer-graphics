from chainer import backend
import chainer.functions as F

def to_homogenous(x):
    """Convert a vector to homogeneous form.

    Args:
        x (:class `~chainer.Variable` or :ref:`ndarray`):
            A 3-D array of shape `(B, M, N)`.

    Returns:
        ~chainer.Variable:
            A 3-D array of shape `(B, M+1, N)`.
    """
    B, M, N = x.shape

    xp = backend.get_array_module(x)
    ones = xp.ones((B, 1, N), dtype=x.dtype)
    return F.concat((x, ones), axis=1)
