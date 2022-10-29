from typing import Optional

import numpy as np
import cvxpy as cp
import scipy

import scipy.linalg as sla

import utils

# nodes = 2
# dim = 5

# bX in variable name means bold X, corresponding to {\bf X} in the article


class Model:
    def __init__(self, nodes, dim, B_rank=1, theta_f=0.9, graph="ring", edge_prob=None):
        self.nodes, self.dim = nodes, dim
        self.theta_f = theta_f

        if graph == "ring":
            self.W = utils.get_ring_W(nodes)  # graph Laplacian
        elif graph == "erdos-renyi":
            self.W = utils.get_ER_W(nodes, edge_prob)

        self.C = np.random.random((nodes, dim, dim))
        self.bC = scipy.linalg.block_diag(*[self.C[i] for i in range(nodes)])

        self.d = [np.random.random(dim) for _ in range(nodes)]
        self.bd = np.hstack(self.d)

        cm = np.random.randint(10, size=(dim, B_rank))
        self.B = np.dot(cm, cm.T)  # Why symmetric ?

        assert np.allclose(self.B.T, self.B)
        beta_min_plus = utils.lambda_min_plus(self.B)  # works if B is symmetric
        w_min_plus = utils.lambda_min_plus(self.W)
        self.gamma_x = beta_min_plus / w_min_plus

        self.gW = (
            self.gamma_x * self.W
        )  # scaling matrix W by gamma (simple preconditioning of A)  # FIXME: dont scale in locally dual approach
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
        self._bCT_bC = None  # used in f_grad
        self._bdT_bC = None  # used in f_grad

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

    @property
    def bCT_bC(self) -> np.ndarray:
        if self._bCT_bC is None:
            self._bCT_bC = self.bC.T @ self.bC
        return self._bCT_bC

    @property
    def bdT_bC(self) -> np.ndarray:
        if self._bdT_bC is None:
            self._bdT_bC = self.bd.T @ self.bC
        return self._bdT_bC

    def get_ATA_lmax_lminp(self):
        lmax = utils.lambda_max(self.B.T @ self.B) + (self.gamma_x * utils.lambda_max(self.W)) ** 2
        lminp = min(utils.lambda_min_plus(self.B.T @ self.B), (self.gamma_x * utils.lambda_min_plus(self.W)) ** 2)
        return lmax, lminp

    def get_mu_L(self):
        L_x = utils.lambda_max(self.bC.T @ self.bC) + self.theta_f
        mu_x = utils.lambda_min(self.bC.T @ self.bC) + self.theta_f

        print("cond", L_x / mu_x)

        lmax, lminp = self.get_ATA_lmax_lminp()
        L_xy = np.sqrt(lmax)
        mu_xy = np.sqrt(lminp)

        return mu_x, mu_xy, L_x, L_xy

    def f(self, bx):
        a = self.bC @ bx - self.bd
        return 1 / 2 * (a.T @ a + self.theta_f * bx.T @ bx)

    def grad_f(self, bx):
        return self.bCT_bC @ bx - self.bdT_bC + self.theta_f * bx

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


def err(bx, model):
    return np.linalg.norm(model.A @ bx)


def crit(bx, model, err_init, f_star, atol=1e-3):
    # return err(bx, model) < err_init * rtol
    return err(bx, model) < atol
    # return abs(model.f(bx) - f_star) / abs(err_init) < rtol

def get_apdm_params(model):
    mu_x, mu_xy, L_x, L_xy = model.get_mu_L()
    delta = (mu_xy**2 / (2 * mu_x * L_x)) ** 0.5

    sigma_x = (mu_x / (2 * L_x)) ** 0.5
    eta_x = min((1 / (4 * (mu_x + L_x * sigma_x))), delta / (4 * L_xy))

    alpha_x = mu_x
    beta_x = 1 / (2 * eta_x * L_xy**2)
    tau_x = 2 * sigma_x / (sigma_x + 0.5)

    x_params = eta_x, alpha_x, beta_x, tau_x, sigma_x

    sigma_y = 1
    eta_y = 1 / (4 * L_xy * delta)

    alpha_y = 0
    tau_y = 2 * sigma_y / (sigma_y + 2)
    beta_y = min(1 / (2 * L_x), 1 / (2 * eta_y * L_xy ** 2))

    rho_b = 1 / max(4 * (1 + (L_x / (2 * mu_x))), 2 * L_xy ** 2 / mu_xy ** 2, 4 * (2 * L_x / mu_x) ** 0.5 * L_xy / mu_xy)
    theta = 1 - rho_b

    y_params = theta, eta_y, alpha_y, beta_y, tau_y, sigma_y
    return x_params, y_params


# def APDG_rate_const(model):
#     """Calculates constant C, such that N <= O(C ln(1/eps)), and what is left under big-O is a constant
#     which is the same for all problems in the class considered"""
#     mu_x, mu_xy, L_x, L_xy = model.get_mu_L()
#     chi_W = utils.lambda_max(model.W) / utils.lambda_min_plus(model.W)
#     BTB = model.B.T @ model.B
#     chi_BTB = utils.lambda_max(BTB) / utils.lambda_min_plus(BTB)
#     mu = utils.lambda_min_plus(model.bW.T @ model.bW) / L_t
#     print(f"Dual problem params: L={L}, mu={mu}, L/mu={L/mu}")


def APDG(iters: int, model: Model, params: Optional[tuple] = None, use_crit=False, accuracy=None) -> (np.ndarray, np.ndarray):
    if params is not None:
        x_params, y_params = params
    else:
        x_params, y_params = get_apdm_params(model)

    eta_x, alpha_x, beta_x, tau_x, sigma_x = x_params
    theta, eta_y, alpha_y, beta_y, tau_y, sigma_y = y_params

    print("eta_x", eta_x, "alpha_x", alpha_x, "beta_x", beta_x, "tau_x", tau_x, "sigma_x", sigma_x)
    print("theta", theta, "eta_y", eta_y, "alpha_y", alpha_y, "beta_y", beta_y, "tau_y", tau_y, "sigma_y", sigma_y)

    # really not required to have these values, but they are appeared to be optimal theoretically
    # assert alpha_y == 0
    # assert sigma_y == 1

    # alpha_y required to be > 0 in the Kovalev's paper, and alpha_y = mu_y. ???
    assert np.all(np.array([eta_x, eta_y, alpha_x, beta_x, beta_y]) > 0)
    assert np.all(0 < np.array([tau_x, tau_y, sigma_x, sigma_y]))
    assert np.all(np.array([tau_x, tau_y, sigma_x, sigma_y]) <= 1)
    assert 0 < theta < 1

    # x_f = x = model.A[0]  # from Range(A^T)
    # y_f = y = yprev = model.A.T[0]  # from Range(A)
    x_f = x = np.zeros(model.nodes * model.dim)

    y_f = y = yprev = np.zeros(model.A.shape[0])

    # logging
    f_err = np.zeros(iters)
    cons_err = np.zeros(iters)
    f_star, x_star = model.solution
    err_init = model.f(x) - f_star
    print("f_star", f_star)

    ATA = model.A.T @ model.A
    AAT = model.A @ model.A.T
    for i in range(iters):
        y_m = y + theta * (y - yprev)

        x_g = tau_x * x + (1 - tau_x) * x_f
        y_g = tau_y * y + (1 - tau_y) * y_f

        df = model.grad_f(x_g)
        dg = 0

        a = alpha_x * (x_g - x)
        b = beta_x * ATA @ x  # -beta_x * model.A.T @ dg
        c = df + model.A.T @ y_m
        xprev = x
        x = x + eta_x * (a - b - c)

        a = 0  # alpha_y * (y_g - y)  # alpha_y = 0
        b = beta_y * (AAT @ y + model.A @ df)
        c = (dg - model.A @ x)
        yprev = y
        y = y + eta_y * (a - b - c)

        x_f = x_g + sigma_x * (x - xprev)
        y_f = y_g + sigma_y * (y - yprev)

        f_err[i] = model.f(x_f) - f_star
        cons_err[i] = np.linalg.norm(model.A @ x)

        if use_crit:
            if crit(x_f, model, err_init, f_star, accuracy):
                break

    return x_f, np.abs(f_err), cons_err


def get_locally_dual_params(model):
    mu_t = utils.lambda_min(model.locally_dual_Q)
    L_t = utils.lambda_max(model.locally_dual_Q)
    L = utils.lambda_max(model.bW.T @ model.bW) / mu_t
    mu = utils.lambda_min_plus(model.bW.T @ model.bW) / L_t
    print(f"Dual problem params: L={L}, mu={mu}, L/mu={L/mu}")

    return 2 / (L + mu)


def LDM(iters: int, model: Model, eta=None, use_crit=False):
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

        if use_crit:
            if crit(bx, model, cons_err[0]):
                break

    return bx, np.abs(f_err), cons_err


def get_globally_dual_params(model):
    mu_x, mu_xy, L_x, L_xy = model.get_mu_L()
    L = utils.lambda_max(model.A.T @ model.A) / mu_x
    mu = utils.lambda_min_plus(model.A.T @ model.A) / L_x
    print(f"Dual problem params: L={L}, mu={mu}, L/mu={L/mu}")

    return 2 / (L + mu)


def GDM(iters: int, model: Model, eta=None, use_crit=False):
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

        if use_crit:
            if crit(bx, model, cons_err[0]):
                break

    return bx, np.abs(f_err), cons_err


def get_globally_dual_accelerated_params(model):
    mu_x, mu_xy, L_x, L_xy = model.get_mu_L()
    L = L_xy ** 2 / mu_x
    mu = mu_xy ** 2 / L_x
    print(f"Dual problem params: L={L}, mu={mu}, L/mu={L/mu}")

    return 1 / L, (L**0.5 - mu**0.5) / (L**0.5 + mu**0.5)


def GDAM(iters: int, model: Model, params: Optional[tuple] = None, use_crit=False, accuracy=None):
    """Globally dual accelerated method (Nesterov's accelerated method)
        To allow for application of Chebyshev acceleration the substitution A^T u -> p, A^T u_ -> q is used
    """
    if params is None:
        eta, beta = get_globally_dual_accelerated_params(model)
    else:
        eta, beta = params

    print(f"globally dual stepsize: {eta}, momentum: {beta}")

    bp = bp_prev = np.zeros(model.A.T.shape[0])

    f_err = np.zeros(iters)
    cons_err = np.zeros(iters)
    f_star, x_star = model.solution
    err_init = model.f(np.zeros(model.dim * model.nodes)) - f_star

    print("f_star", f_star, "err init", err_init)

    ATA = model.A.T @ model.A

    for i in range(iters):
        bq = bp + beta * (bp - bp_prev)
        bx = model.Fstar_grad(bq)
        bp_prev = bp
        bp = bq - eta * ATA @ bx

        f_err[i] = model.f(bx) - f_star
        cons_err[i] = np.linalg.norm(model.A @ bx)

        if use_crit:
            if crit(bx, model, err_init, f_star, accuracy):
                break

    return bx, np.abs(f_err), cons_err


def get_locally_dual_accelerated_params(model):
    mu_t = utils.lambda_min(model.locally_dual_Q)
    L_t = utils.lambda_max(model.locally_dual_Q)
    L = (utils.lambda_max(model.W) * model.gamma_x) ** 2 / mu_t
    mu = (utils.lambda_min_plus(model.W) * model.gamma_x) ** 2 / L_t
    print(f"Dual problem params: L={L}, mu={mu}, L/mu={L / mu}")

    return 1 / L, (L**0.5 - mu**0.5) / (L**0.5 + mu**0.5)


def LDAM(iters: int, model: Model, params: Optional[tuple] = None, use_crit=False, accuracy=None):
    """Locally dual accelerated method (Nesterov accelerated gradient method)"""
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
    err_init = model.f(np.zeros(model.dim * model.nodes)) - f_star

    print("f_star", f_star)

    for i in range(iters):
        bz_ = bz + beta * (bz - bz_prev)
        bt = model.Hstar_grad(bWt.T @ bz_)
        bz_prev = bz
        bz = bz_ - eta * bWt @ bt

        bx = model.bE @ bt
        f_err[i] = model.f(bx) - f_star
        cons_err[i] = np.linalg.norm(bWbE @ bt)

        if use_crit:
            if crit(bx, model, err_init, f_star, accuracy):
                break

    return bx, np.abs(f_err), cons_err

def test_smth():
    iters = 100
    model = Model(nodes=50, dim=20, B_rank=1, graph="erdos-renyi", edge_prob=0.1)
    x_res, f_err, cons_err = APDG(iters, model)
