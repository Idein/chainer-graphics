import numpy as np
from scipy.spatial.transform import Rotation

import chainer_graphics.transform as T

TESTCASES = 100

def test_euler2rot():
    angles = np.random.randn(TESTCASES, 3)
    R1 = Rotation.from_euler('xyz', angles).as_dcm()
    R2 = T.euler2rot(angles).data
    assert(np.allclose(R1, R2))

def test_rot2euler():
    angles1 = np.random.randn(TESTCASES, 3)
    R1 = Rotation.from_euler('xyz', angles1).as_dcm()
    angles2 = T.rot2euler(R1).data
    R2 = Rotation.from_euler('xyz', angles2).as_dcm()
    assert(np.allclose(R1, R2))
