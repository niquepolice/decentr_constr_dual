from typing import Optional

import numpy as np
import cvxpy as cp
import scipy

import scipy.linalg as sla

import networkx as nx
import torch

import utils

# nodes = 2
# dim = 5

# bX in variable name means bold X, corresponding to {\bf X} in the article


class Model:
    def __init__(self, nodes, dim, theta_f=0.9):
        # G = nx.random_graphs.erdos_renyi_graph(9, 1,   directed=False)
        # W = nx.adjacency_matrix(G)
        # D = np.diag(np.sum(np.array(W.todense()), axis=1))
        # L = D - W  # Laplacian
        self.nodes, self.dim = nodes, dim
        self.theta_f = theta_f

        self.W = utils.getW(nodes)  # graph Laplacian

        self.C = torch.rand((nodes, dim, dim))
        self.bC = scipy.linalg.block_diag(*[self.C[i] for i in range(nodes)])

        # self.mC = np.array(bC, dtype=np.float32)
        # self.bC = bC

        self.d = [np.random.random(dim) for _ in range(nodes)]
        self.bd = np.hstack(self.d)

        cm = np.random.randint(10, size=(dim, 1))
        self.B = np.dot(cm, cm.T)  # Why symmetric, rank 1 ?

        beta_min_plus = utils.lambda_min_plus(self.B)
        w_min_plus = utils.lambda_min_plus(self.W)
        gamma_x = beta_min_plus / w_min_plus

        self.gW = (
            gamma_x * self.W
        )  # scaling matrix W by gamma (simple preconditioning of A)
        Im = np.identity(nodes)
        Id = np.identity(dim)
        self.bB, self.bW = np.kron(Im, self.B), np.kron(self.gW, Id)
        self.A = np.vstack((self.bB, self.bW))

        self.solution = self._get_solution()

        # retained calculations TODO: move to callable class LDGD
        self._E = None  # used in locally dual method
        self._bE = None  # used in locally dual method
        self._bdT_bC_bE = None  # used in H_grad
        self._locally_dual_Q = None  # used in H_grad

    @property
    def E(self) -> np.ndarray:
        if self._E is None:
            self._E = scipy.linalg.null_space(self.bB[: self.dim, : self.dim])
        return self._E

    @property
    def bE(self) -> np.ndarray:
        if self._bE is None:
            self._bE = scipy.linalg.block_diag(*[self.E for _ in range(self.nodes)])
        return self._bE

    @property
    def bdT_bC_bE(self) -> np.ndarray:
        if self._bdT_bC_bE is None:
            self._bdT_bC_bE = self.bd.T @ self.bC @ self.bE
        return self._bdT_bC_bE

    @property
    def locally_dual_Q(self) -> np.ndarray:
        if self._locally_dual_Q is None:
            I = np.identity(self.bC.shape[0])
            self._locally_dual_Q = (
                self.bE.T @ (self.bC.T @ self.bC + self.theta_f * I) @ self.bE
            )
        return self._locally_dual_Q

    def get_mu_L(self):
        L_x = utils.lambda_max(self.bC.T @ self.bC) + self.theta_f
        mu_x = utils.lambda_min(self.bC.T @ self.bC) + self.theta_f

        print("cond", L_x / mu_x)

        L_xy = np.sqrt(utils.lambda_max(self.A.T @ self.A))
        mu_xy = np.sqrt(utils.lambda_min_plus(self.A @ self.A.T))

        return mu_x, mu_xy, L_x, L_xy

    def f(self, bx):
        if type(bx) == np.ndarray:
            bx = torch.from_numpy(bx).float()
        _bC = torch.from_numpy(self.bC)
        _bC.requires_grad_(True)
        _bd = torch.from_numpy(self.bd).float()

        a = _bC @ bx - _bd
        return 1 / 2 * (a.T @ a + self.theta_f * bx.T @ bx)

    def grad_f(self, bx):
        bx = torch.from_numpy(bx).float()
        bx.requires_grad_(True)
        f_val = self.f(bx)
        f_val.backward()
        return bx.grad.numpy()

    def _get_solution(self):
        x = cp.Variable(self.dim)
        # cp.Minimize( (1/2)*cp.quad_form(x, C) + q.T @ x ),
        Q = sum(
            self.C[i].T @ self.C[i] + self.theta_f * np.identity(self.dim)
            for i in range(self.nodes)
        )
        q = sum(self.d[i].T @ np.array(self.C[i]) for i in range(self.nodes))
        const = 1 / 2 * sum(self.d[i].T @ self.d[i] for i in range(self.nodes))

        prob = cp.Problem(
            cp.Minimize(1 / 2 * cp.quad_form(x, Q) - q @ x + const), [self.B @ x == 0]
        )

        prob.solve()

        return prob.value, x.value

    def Fstar_grad(self, bu):
        # -(z.T @ t - F(x)) -> min_x
        I = np.identity(self.bC.shape[0])
        return np.linalg.solve(
            self.bC.T @ self.bC + self.theta_f * I, self.bd.T @ self.bC + bu
        )

    def Hstar_grad(self, bz):

        # -(z.T @ t - F(Et)) -> min_t
        return np.linalg.solve(self.locally_dual_Q, self.bdT_bC_bE.T + bz)


def get_apdm_params(model):
    mu_x, mu_xy, L_x, L_xy = model.get_mu_L()
    delta = (mu_xy**2 / (2 * mu_x * L_x)) ** 0.5

    sigma_x = (mu_x / (2 * L_x)) ** 0.5
    eta_x = min((1 / (4 * (mu_x + L_x * sigma_x))), delta / (4 * L_xy))

    alpha_x = mu_x
    beta_x = 1 / (2 * eta_x * L_xy**2)
    tau_x = 2 * sigma_x / (sigma_x + 0.5)

    return eta_x, alpha_x, beta_x, tau_x, sigma_x


def APDM(iters: int, model: Model, params: Optional[tuple] = None) -> (np.ndarray, np.ndarray):
    if params is not None:
        eta_x, alpha_x, beta_x, tau_x, sigma_x = params
    else:
        eta_x, alpha_x, beta_x, tau_x, sigma_x = get_apdm_params(model)

    print("eta_x", eta_x, "alpha_x", alpha_x, "beta_x", beta_x, "tau_x", tau_x, "sigma_x", sigma_x)
    x = model.A[0]  # from Range(A^T)
    # x = np.zeros(model.nodes * model.dim)

    x_f = x.copy()
    # x_f = np.zeros(model.nodes * model.dim)

    # logging
    f_err = np.zeros(iters)
    cons_err = np.zeros(iters)
    f_star, x_star = model.solution
    print("f_star", f_star)

    ATA = model.A.T @ model.A
    for i in range(iters):
        x_g = tau_x * x + (1 - tau_x) * x_f
        a = alpha_x * (x_g - x)
        b = beta_x * ATA @ x
        c = model.grad_f(x_g)
        xprev = x
        x = x + eta_x * (a - b - c)
        x_f = x_g + sigma_x * (x - xprev)

        f_err[i] = model.f(x_f) - f_star
        cons_err[i] = np.linalg.norm(model.A @ x)

    return x_f, np.abs(f_err), cons_err


def get_locally_dual_params(model):
    mu_t = utils.lambda_min(model.locally_dual_Q)
    L_t = utils.lambda_max(model.locally_dual_Q)
    L = utils.lambda_max(model.bW.T @ model.bW) / mu_t
    mu = utils.lambda_min_plus(model.bW.T @ model.bW) / L_t
    print(f"Dual problem params: L={L}, mu={mu}, L/mu={L/mu}")

    return 2 / (L + mu)


def LDM(iters: int, model: Model, eta=None):
    """Locally dual nonaccelerated method"""
    if eta is None:
        eta = get_locally_dual_params(model)
    print("locally dual stepsize:", eta)

    bWt = np.kron(model.gW, np.identity(model.E.shape[1]))
    bWbE = model.bW @ model.bE

    bz = np.zeros(bWt.shape[0])

    f_err = np.zeros(iters)
    cons_err = np.zeros(iters)
    f_star, x_star = model.solution

    print("f_star", f_star)

    for i in range(iters):
        bt = model.Hstar_grad(bWt.T @ bz)
        bz = bz - eta * bWt @ bt

        bx = model.bE @ bt
        f_err[i] = model.f(bx) - f_star
        cons_err[i] = np.linalg.norm(bWbE @ bt)

    return bx, np.abs(f_err), cons_err


def get_globally_dual_params(model):
    mu_x, mu_xy, L_x, L_xy = model.get_mu_L()
    L = utils.lambda_max(model.A.T @ model.A) / mu_x
    mu = utils.lambda_min_plus(model.A.T @ model.A) / L_x
    print(f"Dual problem params: L={L}, mu={mu}, L/mu={L/mu}")

    return 2 / (L + mu)


def GDM(iters: int, model: Model, eta=None):
    """Globally dual nonaccelerated method"""
    if eta is None:
        eta = get_globally_dual_params(model)
    print("globally dual stepsize:", eta)

    bu = np.zeros(model.bB.shape[0] + model.bW.shape[0])

    f_err = np.zeros(iters)
    cons_err = np.zeros(iters)
    f_star, x_star = model.solution

    print("f_star", f_star)

    for i in range(iters):
        bx = model.Fstar_grad(model.A.T @ bu)
        bu = bu - eta * model.A @ bx

        f_err[i] = model.f(bx) - f_star
        cons_err[i] = np.linalg.norm(model.A @ bx)

    return bx, np.abs(f_err), cons_err


def get_globally_dual_accelerated_params(model):
    mu_x, mu_xy, L_x, L_xy = model.get_mu_L()
    L = utils.lambda_max(model.A.T @ model.A) / mu_x
    mu = utils.lambda_min_plus(model.A.T @ model.A) / L_x
    print(f"Dual problem params: L={L}, mu={mu}, L/mu={L/mu}")

    return 1 / L, (L**0.5 - mu**0.5) / (L**0.5 + mu**0.5)


def GDAM(iters: int, model: Model, params: Optional[tuple] = None):
    """Globally dual accelerated method (Nesterov's accelerated method)"""
    if params is None:
        eta, beta = get_globally_dual_accelerated_params(model)
    else:
        eta, beta = params

    print(f"globally dual stepsize: {eta}, momentum: {beta}")

    bu = np.zeros(model.bB.shape[0] + model.bW.shape[0])
    bu_prev = bu.copy()

    f_err = np.zeros(iters)
    cons_err = np.zeros(iters)
    f_star, x_star = model.solution

    print("f_star", f_star)

    for i in range(iters):
        bu_ = bu + beta * (bu - bu_prev)
        bx = model.Fstar_grad(model.A.T @ bu_)
        bu_prev = bu
        bu = bu_ - eta * model.A @ bx

        f_err[i] = model.f(bx) - f_star
        cons_err[i] = np.linalg.norm(model.A @ bx)

    return bx, np.abs(f_err), cons_err


def get_locally_dual_accelerated_params(model):
    mu_t = utils.lambda_min(model.locally_dual_Q)
    L_t = utils.lambda_max(model.locally_dual_Q)
    L = utils.lambda_max(model.bW.T @ model.bW) / mu_t
    mu = utils.lambda_min_plus(model.bW.T @ model.bW) / L_t
    print(f"Dual problem params: L={L}, mu={mu}, L/mu={L / mu}")

    return 1 / L, (L**0.5 - mu**0.5) / (L**0.5 + mu**0.5)


def LDAM(iters: int, model: Model, params: Optional[tuple] = None):
    """Locally dual accelerated method (Neseterov accelerated gradient method)"""
    if params is None:
        eta, beta = get_locally_dual_accelerated_params(model)
    else:
        eta, beta = params

    print(f"locally dual stepsize: {eta}, momentum: {beta}")

    bWt = np.kron(model.gW, np.identity(model.E.shape[1]))
    bWbE = model.bW @ model.bE

    bz = np.zeros(bWt.shape[0])
    bz_prev = bz.copy()

    f_err = np.zeros(iters)
    cons_err = np.zeros(iters)
    f_star, x_star = model.solution

    print("f_star", f_star)

    for i in range(iters):
        bz_ = bz + beta * (bz - bz_prev)
        bt = model.Hstar_grad(bWt.T @ bz_)
        bz_prev = bz
        bz = bz_ - eta * bWt @ bt

        bx = model.bE @ bt
        f_err[i] = model.f(bx) - f_star
        cons_err[i] = np.linalg.norm(bWbE @ bt)

    return bx, np.abs(f_err), cons_err
