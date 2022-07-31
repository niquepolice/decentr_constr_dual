import numpy as np


def lambda_max(M):
    eigvals, eigvecs = np.linalg.eigh(M)
    # lambda_max = max(zna)
    return eigvals[-1]


def lambda_min(M):
    eigvals, eigvecs = np.linalg.eigh(M)
    # zna, vek = np.linalg.eigh(M)
    # lambda_min = min(zna)
    return eigvals[0]


def lambda_min_plus(M):
    """Returns minimal positive eigenvalue"""
    eigvals, eigvecs = np.linalg.eigh(M)
    lambda_min_plus = min([n for n in eigvals if n > 1e-6])
    return lambda_min_plus


def getW(nodes: int) -> np.ndarray:
    """Returns Laplacian of the ring graph"""
    if nodes == 1:
        return np.array([[0]])
    if nodes == 2:
        return np.array([[1, -1], [-1, 1]])
    w1 = np.zeros(nodes)
    w1[0], w1[-1], w1[1] = 2, -1, -1
    W = np.array([np.roll(w1, i) for i in range(nodes)])
    return W
