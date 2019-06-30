from chainer import backend
import chainer.functions as F

def distort_points(coef, x):
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
    if K < 8:
        coef = F.pad(coef, ((0,0), (0,8-K)), 'constant')
    coef = coef[:, :, None]


    # Compute
    # f = (1 + k1r^2 + k2r^4 + k3r^6) / (1 + k4r^2 + k5r^4 + k6r^6)
    r2 = F.sum(x * x, 1, keepdims=True)  # r^2
    f = (1 + r2 * (coef[:, 0:1] + r2 * (coef[:, 1:2] + r2 * coef[:, 4:5]))) / \
        (1 + r2 * (coef[:, 5:6] + r2 * (coef[:, 6:7] + r2 * coef[:, 7:8])))

    xy = F.prod(x, 1, keepdims=True)

    return x * f + 2 * xy * coef[:, 2:4] + coef[:, 3:1:-1] * (r2 + 2 * x * x)

def undistort_points(coef, p, iteration=5):
    """Remove distortion from given points.

    Args:
        coef (:class `~chainer.Variable` or :ref:`ndarray`):
            Distortion coefficients.
            A 2-D array of shape `(B, K)`
            K is 4 or 5 or 8. The elements corresponds to
            (k1, k2, p1, p2, [k3, [k4, k5 k6]])
            respectively.

        p (:class `~chainer.Variable` or :ref:`ndarray`):
            A 3-D array of shape `(B, 2, N)`
            
    Returns:
        ~chainer.Variable:
            A 3-D array of shape `(B, 2, N)`
    """

    xp = backend.get_array_module(p)
    B, _, N = p.shape
    _, K = coef.shape
    if K < 8:
        coef = F.pad(coef, ((0,0), (0,8-K)), 'constant')

    # (B, 8) -> (B, 1, 8)
    coef = coef[:, None, :]

    k1 = coef[:, :, 0:1]
    k2 = coef[:, :, 1:2]
    p1 = coef[:, :, 2:3]
    p2 = coef[:, :, 3:4]
    k3 = coef[:, :, 4:5]
    k4 = coef[:, :, 5:6]
    k5 = coef[:, :, 6:7]
    k6 = coef[:, :, 7:8]

    # (B, 2, N) -> (B, N, 2)
    p = p.transpose((0, 2, 1))
    r2 = F.sum(p * p, 2, keepdims=True) # r^2

    # Compute initial guess
    X = (1 - r2 * (k1 + r2 * (3*k1**2 - k2 + r2 * (8*k1*k2 - 12*k1**3 - k3)))) * p

    # Refinement by Newton-Raphson method
    for i in range(iteration):
        x = X[:, :, 0:1]
        y = X[:, :, 1:2]
        xy = F.prod(X, 2, keepdims=True)
        r2 = F.sum(X * X, 2, keepdims=True)  # r^2
        a = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        b = 1 + r2 * (k4 + r2 * (k5 + r2 * k6))
        da = k1 + r2 * (2 * k2 + 3 * r2 * k3)
        db = k4 + r2 * (2 * k5 + 3 * r2 * k6)

        g = a/b
        dg = (da*b - a*db) / b**2

        J00 = g + 2*x**2*dg + 2*y*p1 + 6*x*p2
        J11 = g + 2*y**2*dg + 2*x*p2 + 6*y*p1
        J01 = 2*F.prod(x, 2, keepdims=True)*dg + 2*x*p1 + 2*y*p2
        jacobian = F.stack([J00, J01, J01, J11], axis=2).reshape(B, N, 2, 2)

        d = g * X + 2 * xy * coef[:, :, 2:4] + coef[:, :, 3:1:-1] * (r2 + 2 * X * X)

        jacobian = jacobian.reshape(-1, 2, 2)
        f = (d - p).reshape(-1, 2)
        X = X - F.batch_matmul(F.batch_inv(jacobian), f).reshape(B, N, 2)

    # (B, N, 2) -> (B, 2, N)
    return X.transpose((0, 2, 1))
