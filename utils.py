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
    tol = 1e-6

    lam_min_plus = eigvals[eigvals > tol].min()
    small_nonzero = eigvals[(eigvals <= tol) & (eigvals > 0)]
    # if small_nonzero.size > 0:
    #     print("note: small nonzero eigenvals interpreted as 0:", small_nonzero)
    return lam_min_plus


def get_ring_W(nodes: int) -> np.ndarray:
    """Returns Laplacian of the ring graph"""
    if nodes == 1:
        return np.array([[0]])
    if nodes == 2:
        return np.array([[1, -1], [-1, 1]])
    w1 = np.zeros(nodes)
    w1[0], w1[-1], w1[1] = 2, -1, -1
    W = np.array([np.roll(w1, i) for i in range(nodes)])
    return W


def get_ER_W(nodes: int, p: float) -> np.ndarray:
    """Returns Laplacian of the Erdos-Renyi graph"""
    import networkx as nx

    graph = nx.random_graphs.erdos_renyi_graph(nodes, p, directed=False, seed=np.random)
    M = nx.to_numpy_array(graph)
    D = np.diag(np.sum(M, axis=1))
    print("mean degree:", D.sum() / nodes)
    W = D - M  # Laplacian
    return W


def test_ER():
    assert np.all(get_ER_W(1, 1) == get_ring_W(1))
    assert np.all(get_ER_W(2, 1) == get_ring_W(2))
    assert np.all(get_ER_W(3, 1) == get_ring_W(3))

    for n in range(2, 10):
        for _ in range(5):
            p = np.random.random()
            W = get_ER_W(n, p)
            assert W.shape == (n, n)
            assert np.all(W.T == W)
            assert np.all(W @ np.ones(n) == np.zeros(n))
            assert np.all((W >= 0) | (W == -1))

