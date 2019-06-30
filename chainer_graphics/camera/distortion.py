from chainer import backend

def distort(coef, x):
    """Apply distortion to given points.

    Args:
        coef (:class `~chainer.Variable` or :ref:`ndarray`):
            Distortion coefficients.
            A 2-D array of shape `(B, K)`
            K is 4 or 5 or 8. The elements corresponds to
            (k1, k2, p1, p2, [k3, [k4, k5 k6]])
            respectively.

        x (:class `~chainer.Variable` or :ref:`ndarray`):
            A 3-D array of shape `(B, 2, N)`
            
    Returns:
        ~chainer.Variable:
            A 3-D array of shape `(B, 2, N)`
    """
    xp = backend.get_array_module(x)
    _, K = coef.shape

    coef = coef[:, :, None]

    r2 = (x * x).sum(1, keepdims=True)  # r^2

    # Compute
    # f = (1 + k1r^2 + k2r^4 + k3r^6) / (1 + k4r^2 + k5r^4 + k6r^6)
    r4 = r2 * r2
    r6 = r4 * r2
    f = 1 + r2 * coef[:, 0:1] + r4 * coef[:, 1:2]
    if K > 4:
        f += r6 * coef[:, 4:5]
    if K > 5:
        f = f / (1 + r2 * coef[:, 5:6] + r4 * coef[:, 6:7] + r6 * coef[:, 7:8])

    xy = x.prod(1, keepdims=True)

    return x * f + 2 * xy * coef[:, 2:4] + coef[:, 4:1:-1] * (r2 + 2 * x * x)
