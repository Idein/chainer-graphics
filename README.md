# chainer-graphics

Differential Graphics Operations for chainer.

Currently implemented operations.

- `chainer_graphics.camera`
  - Projection and Re-projection
  - Camera matrix
  - Distortion
- `chainer_graphics.image`
  - Image warping
- `chainer_graphics.transform`
  - Euler angles

# Requirements

- Python >= 3.5
- Chainer >= 6.0.0

# Installation

```
$ pip install git+https://github.com/Idein/chainer-graphics
```

# Tests

```
$ nosetests -v
```
