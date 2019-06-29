from chainer import backend
import chainer.functions as F

def pixel_coords(xp, H, W, dtype):
    us, vs = xp.meshgrid(xp.arange(W).astype(dtype), xp.arange(H).astype(dtype))
    ps = F.stack((us, vs), axis=0)
    return ps

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

    A = mat[:, :, :2]
    Ainv = F.batch_inv(A)
    t = mat[:, :, 2].reshape(-1, 2, 1)
    ps1 = pixel_coords(xp, H, W, mat.dtype).reshape(1, 2, -1)

    ps0 = Ainv @ (ps1 - t)
    ps0 = 2 * ps0 / xp.array([W-1, H-1]).reshape(-1, 2, 1) - 1
    ps0 = ps0.reshape(B, 2, H, W)

    warped_image = F.spatial_transformer_sampler(image, ps0)
    return warped_image

def warp_perspective(image, mat):
    """Warp images with perspective transformation

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

    pixel_coords(xp, H, W, mat.dtype) 

def warp_dense(image):
    #x -> x'
    #cam2pixel(x) -> cam2pixel(x')
    #pass
    pass
