import chainer.functions as F
def pixel_coords(xp, H, W, dtype):
    """Generate pixel coordinates"""
    us, vs = xp.meshgrid(xp.arange(W).astype(dtype), xp.arange(H).astype(dtype))
    ps = xp.stack((us, vs), axis=0)
    return ps

