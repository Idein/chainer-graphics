import chainer.functions as F

def cam2pixel(focal, offset, x):
    """Convert 3D points to pixel coordinates.

    Args:
        focal (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 2)`.
            Focal length fx and fy.

        offse (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 2)`.
            Principal point offsets cx and cy.

        x (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 3)`.
            3D points x, y and z.

    Returns:
        ~chainer.Variable:
            A 2-D array of shape `(B, 2)`.
            Pixel coordinates u and v.
    """

    u = focal[:,0] * x[:,0] / x[:,2] + offset[:,0]
    v = focal[:,1] * x[:,1] / x[:,2] + offset[:,1]
    return F.stack((u, v), axis=1)

def pixel2cam(focal, offset, p, z):
    """Convert pixel coordinates to 3D points

    Args:
        focal (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 2)`.
            Focal length fx and fy.

        offse (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 2)`.
            Principal point offsets cx and cy.

        p (:class `~chainer.Variable` or :ref:`ndarray`):
            A 2-D array of shape `(B, 2)`.
            Pixel coordinates u and v.

        z (:class `~chainer.Variable` or :ref:`ndarray`):
            A 1-D array of length `B`.
            z-coordinates of corresponding (u, v).

    Returns:
        ~chainer.Variable:
            A 2-D array of shape `(B, 3)`.
            3D points x, y and z.
    """

    x = (z * p[:,0] - offset[:, 0]) / focal[:, 0]
    y = (z * p[:,1] - offset[:, 1]) / focal[:, 1]
    return F.stack((x, y, z), axis=1)
