# -*- coding: utf-8 -*-
"""
T09：一维油水两相渗流，全隐式 Newton。

未知量按单元交错排列：u = [p_0, S_0, p_1, S_1, ...]
p 为油相压力，S 为含水饱和度。

边界：
- 左端 x=0：定总流量注入，注入水相分数 fw_inj
- 右端 x=L：定压 p0，出流时饱和度零梯度，入流时取初始含水饱和度

Jacobian 用局部残差的有限差分组装，利用残差只依赖相邻三单元的结构。
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .grid import make_grid
from .timestepping import ConvergenceError, advance


# ------------------------------------------------------------------ 物性
def _Sd(S, prm):
    denom = 1.0 - prm["Swr"] - prm["Sor"]
    return np.clip((S - prm["Swr"]) / denom, 1e-5, 1.0 - 1e-5)


def krw(S, prm):
    return prm["krw0"] * _Sd(S, prm) ** prm["nw"]


def kro(S, prm):
    return prm["kro0"] * (1.0 - _Sd(S, prm)) ** prm["no"]


def pc(S, prm):
    """毛管压力，Pa。J(S)=Sd^-0.5。"""
    return prm["Ac"] * _Sd(S, prm) ** (-0.5)


def phi_of(p, prm):
    return prm["phi0"] * (1.0 + prm["cphi"] * (p - prm["p0"]))


def rho_w_of(pw, prm):
    return prm["rho_w"] * (1.0 + prm["cw"] * (pw - prm["p0"]))


def rho_o_of(p, prm):
    return prm["rho_o"] * (1.0 + prm["co"] * (p - prm["p0"]))


def _mobility_w(S, prm):
    return prm["k"] * krw(S, prm) / prm["mu_w"]


def _mobility_o(S, prm):
    return prm["k"] * kro(S, prm) / prm["mu_o"]


# ------------------------------------------------------------------ 参数
def build_prm(case=0, fw_inj=0.5, p_inj_max=None, N_f=10):
    import config
    c = config.case_params(case)
    return {
        "phi0": c["φ_0"], "cphi": c["c_φ"], "k": c["k_m0"],
        "rho_o": c["ρ_o"], "rho_w": c["ρ_w"],
        "rho_w_ref": 1000.0, "rho_o_ref": c["ρ_o"],
        "mu_o": c["μ_o"], "mu_w": c["μ_w"],
        "co": c["c_o"], "cw": c["c_w"],
        "Sw0": c["S_w0"], "Swr": c["S_wr"], "Sor": c["S_or"],
        "kro0": c["k_ro0"], "krw0": c["k_rw0"],
        "no": c["n_o"], "nw": c["n_w"], "Ac": c["A_c"],
        "Lf": c["L_f"], "hf": c["h_f"], "qinj": c["q_inj"],
        "mu_co2": c["μ_CO2"],
        "p0": c["p_0"], "fw_inj": fw_inj,
        "p_max": p_inj_max if p_inj_max is not None else c["p_max"],
        "N_f": N_f,
    }


# ------------------------------------------------------------------ 通量
def _face_flux(pl, Sl, pr, Sr, dxf, prm):
    """正方向为 +x，返回 (Fw, Fo)，单位 kg/(m2 s)。"""
    pwl = pl - pc(Sl, prm)
    pwr = pr - pc(Sr, prm)

    if pwl >= pwr:
        lam_w, rw = _mobility_w(Sl, prm), rho_w_of(pwl, prm)
    else:
        lam_w, rw = _mobility_w(Sr, prm), rho_w_of(pwr, prm)
    Fw = -rw * lam_w * (pwr - pwl) / dxf

    if pl >= pr:
        lam_o, ro = _mobility_o(Sl, prm), rho_o_of(pl, prm)
    else:
        lam_o, ro = _mobility_o(Sr, prm), rho_o_of(pr, prm)
    Fo = -ro * lam_o * (pr - pl) / dxf
    return Fw, Fo


class TwoPhase:
    def __init__(self, L=40.0, n=200, stretch=0.5, case=0,
                 fw_inj=0.5, p_end=None, q_inj_ref=None, N_f=10):
        self.prm = build_prm(case, fw_inj, N_f=N_f)
        if q_inj_ref is not None:
            self.prm["qinj"] = q_inj_ref
        self.prm["p_end"] = self.prm["p0"] if p_end is None else p_end

        self.xf, self.xc, self.dx = make_grid(L, n, stretch=stretch)
        self.N = len(self.dx)
        self.dxf = 0.5 * (self.dx[:-1] + self.dx[1:])

        self.P = np.full(self.N, self.prm["p0"])
        self.S = np.full(self.N, self.prm["Sw0"])
        self.t = 0.0

        # 注入速度：config 已把 m3/d 转成 m3/s，这里只需除以面积
        area = self.prm["N_f"] * 4.0 * self.prm["Lf"] * self.prm["hf"]
        self.v_inj = self.prm["qinj"] / area

        self.mass_in_w = 0.0
        self.mass_in_o = 0.0
        self.mass_out_w = 0.0
        self.mass_out_o = 0.0
        self.mw0, self.mo0 = self._stored()

    # -------------------------------------------------------------- 状态
    def _stored(self):
        prm = self.prm
        ph = phi_of(self.P, prm)
        pw = self.P - pc(self.S, prm)
        mw = np.sum(self.dx * ph * rho_w_of(pw, prm) * self.S)
        mo = np.sum(self.dx * ph * rho_o_of(self.P, prm) * (1 - self.S))
        return float(mw), float(mo)

    def pack(self):
        u = np.empty(2 * self.N)
        u[0::2] = self.P
        u[1::2] = self.S
        return u

    def unpack(self, u):
        self.P = u[0::2].copy()
        self.S = u[1::2].copy()

    # -------------------------------------------------------------- 单步残差
    def _cell_res(self, i, P, S, Pold, Sold, dt):
        prm = self.prm
        dx = self.dx[i]
        ph = phi_of(P[i], prm)
        pw = P[i] - pc(S[i], prm)
        Aw = ph * rho_w_of(pw, prm) * S[i]
        Ao = ph * rho_o_of(P[i], prm) * (1.0 - S[i])

        ph0 = phi_of(Pold[i], prm)
        pw0 = Pold[i] - pc(Sold[i], prm)
        Aw0 = ph0 * rho_w_of(pw0, prm) * Sold[i]
        Ao0 = ph0 * rho_o_of(Pold[i], prm) * (1.0 - Sold[i])

        # 左界面
        if i == 0:
            FwL, FoL = self._injection_flux()
        else:
            FwL, FoL = _face_flux(P[i - 1], S[i - 1], P[i], S[i],
                                  self.dxf[i - 1], prm)

        # 右界面
        if i == self.N - 1:
            FwR, FoR = self._right_boundary(P[i], S[i])
        else:
            FwR, FoR = _face_flux(P[i], S[i], P[i + 1], S[i + 1],
                                  self.dxf[i], prm)

        Rw = dx * (Aw - Aw0) / dt + FwR - FwL
        Ro = dx * (Ao - Ao0) / dt + FoR - FoL
        return Rw, Ro

    def _injection_flux(self):
        prm = self.prm
        return (prm["rho_w_ref"] * self.v_inj * prm["fw_inj"],
                prm["rho_o_ref"] * self.v_inj * (1.0 - prm["fw_inj"]))

    def _right_boundary(self, pressure, saturation):
        prm = self.prm
        boundary_saturation = saturation if pressure >= prm["p_end"] else prm["Sw0"]
        return _face_flux(pressure, saturation, prm["p_end"], boundary_saturation,
                          0.5 * self.dx[-1], prm)

    def residual(self, u, uold, dt):
        P, S = u[0::2], u[1::2]
        Pold, Sold = uold[0::2], uold[1::2]
        R = np.empty_like(u)
        for i in range(self.N):
            Rw, Ro = self._cell_res(i, P, S, Pold, Sold, dt)
            R[2 * i] = Rw
            R[2 * i + 1] = Ro
        return R

    # -------------------------------------------------------------- Jacobian
    def jacobian(self, u, uold, dt):
        N = self.N
        eps_p = 1e-6 * np.maximum(np.abs(u[0::2]), 1e5)
        eps_s = 1e-7
        rows, cols, vals = [], [], []

        base_R = np.empty((N, 2))
        for i in range(N):
            base_R[i] = self._cell_res(i, u[0::2], u[1::2],
                                       uold[0::2], uold[1::2], dt)

        for i in range(N):
            lo = max(0, i - 1)
            hi = min(N - 1, i + 1)
            for j in range(lo, hi + 1):
                for var in (0, 1):
                    k = 2 * j + var
                    eps = eps_p[j] if var == 0 else eps_s
                    u[k] += eps
                    Rp = np.array(self._cell_res(i, u[0::2], u[1::2],
                                                 uold[0::2], uold[1::2], dt))
                    u[k] -= eps
                    dR = (Rp - base_R[i]) / eps
                    rows += [2 * i, 2 * i + 1]
                    cols += [k, k]
                    vals += [dR[0], dR[1]]

        return coo_matrix((vals, (rows, cols)), shape=(2 * N, 2 * N)).tocsr()

    # -------------------------------------------------------------- Newton
    def step(self, dt_s, maxit=20, tol=1e-8):
        """dt_s: 时间步，秒。"""
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("Time step must be positive and finite")
        u = self.pack()
        uold = u.copy()
        Pold, Sold = uold[0::2].copy(), uold[1::2].copy()

        scale = np.empty_like(u)
        for i in range(self.N):
            s = self.dx[i] * 1000.0 / dt_s
            scale[2 * i] = scale[2 * i + 1] = s

        R = self.residual(u, uold, dt_s)
        err0 = np.max(np.abs(R / scale))
        err = err0
        it = 0
        for it in range(1, maxit + 1):
            if err < tol:
                break
            J = self.jacobian(u, uold, dt_s)
            du = spsolve(J, -R)
            if not np.all(np.isfinite(du)):
                raise ConvergenceError("Jacobian solve produced non-finite update")

            alpha = 1.0
            for _ in range(8):
                un = u + alpha * du
                un[1::2] = np.clip(un[1::2], 1e-6, 1.0 - 1e-6)
                un[0::2] = np.clip(un[0::2], 1e5, 1e8)
                Rn = self.residual(un, uold, dt_s)
                err_n = np.max(np.abs(Rn / scale))
                if err_n < err or err_n < tol:
                    break
                alpha *= 0.5
            else:
                raise ConvergenceError(f"Newton line search stalled at residual {err:.2e}")

            u, R, err = un, Rn, err_n
            if err < tol:
                break

        if not np.isfinite(err) or err >= tol:
            raise ConvergenceError(f"Newton residual {err:.2e} exceeds tolerance {tol:.2e}")

        self.unpack(u)
        self.t += dt_s / 86400.0

        water_in, oil_in = self._injection_flux()
        water_out, oil_out = self._right_boundary(self.P[-1], self.S[-1])
        self.mass_in_w += water_in * dt_s
        self.mass_in_o += oil_in * dt_s
        self.mass_out_w += water_out * dt_s
        self.mass_out_o += oil_out * dt_s
        return it, err

    def run(self, t_end, dt, maxit=20, tol=1e-8, verbose=False):
        """t_end 与 dt 均以天计。"""
        nstep = int(round(t_end / dt))
        hist = []
        for k in range(nstep):
            it, err = advance(self, dt * 86400.0, maxit=maxit, tol=tol)
            if verbose and (k % max(1, nstep // 10) == 0):
                print(f"  step {k}/{nstep}, it={it}, err={err:.2e}")
            hist.append((self.t, self.P.copy(), self.S.copy()))
        return hist

    # -------------------------------------------------------------- 守恒
    def mass_balance(self):
        prm = self.prm
        ph = phi_of(self.P, prm)
        pw = self.P - pc(self.S, prm)
        mw = np.sum(self.dx * ph * rho_w_of(pw, prm) * self.S)
        mo = np.sum(self.dx * ph * rho_o_of(self.P, prm) * (1 - self.S))
        return {
            "water_in": self.mass_in_w,
            "water_out": self.mass_out_w,
            "water_stored": mw,
            "water_initial": self.mw0,
            "water_residual": self.mw0 + self.mass_in_w - self.mass_out_w - mw,
            "oil_in": self.mass_in_o,
            "oil_out": self.mass_out_o,
            "oil_stored": mo,
            "oil_initial": self.mo0,
            "oil_residual": self.mo0 + self.mass_in_o - self.mass_out_o - mo,
        }
