import numpy as np

eps = 1e-3

def cosine_similarity(image0, image1):
    v0 = image0.flatten()
    v1 = image1.flatten()
    return v0.dot(v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
