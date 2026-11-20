# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
"""
T10：一维水/油/气三相渗流（自由 CO₂ 相），全隐式 Newton。

未知量按单元交错排列：u = [p, S_w, S_g] * N
p 为油相压力；S_o = 1 - S_w - S_g。
气相压力取 p_g = p_o（忽略气-油毛管），水-油毛管沿用两相模型。

本版先做不可溶气体，用于验证注入能力；溶解与膨胀在后续版本加入。
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .grid import make_grid
from .timestepping import ConvergenceError, advance


# ------------------------------------------------------------------ 物性
def _denom(prm):
    return max(1.0 - prm["Swr"] - prm["Sor"] - prm["Sgc"], 1e-6)


def _Sd(Sw, prm):
    return np.clip((Sw - prm["Swr"]) / _denom(prm), 1e-6, 1.0 - 1e-6)


def _Sgd(Sg, prm):
    return np.clip((Sg - prm["Sgc"]) / _denom(prm), 1e-6, 1.0 - 1e-6)


def _Sod(Sw, Sg, prm):
    v = 1.0 - (Sw - prm["Swr"]) / _denom(prm) - (Sg - prm["Sgc"]) / _denom(prm)
    return np.clip(v, 1e-6, 1.0 - 1e-6)


def krw(Sw, prm):
    return prm["krw0"] * _Sd(Sw, prm) ** prm["nw"]


def kro(Sw, Sg, prm):
    return prm["kro0"] * _Sod(Sw, Sg, prm) ** prm["no"]


def krg(Sg, prm):
    return prm["krg0"] * _Sgd(Sg, prm) ** prm["ng"]


def pc(Sw, prm):
    return prm["Ac"] * _Sd(Sw, prm) ** (-0.5)


def phi_of(p, prm):
    return prm["phi0"] * (1.0 + prm["cphi"] * (p - prm["p0"]))


def rho_w_of(pw, prm):
    return prm["rho_w"] * (1.0 + prm["cw"] * (pw - prm["p0"]))


def rho_o_of(p, prm):
    return prm["rho_o"] * (1.0 + prm["co"] * (p - prm["p0"]))


def rho_g_of(p, prm):
    return prm["rho_g"] * (1.0 + prm["cg"] * (p - prm["p0"]))


def _lam_w(Sw, prm):
    return prm["k"] * krw(Sw, prm) / prm["mu_w"]


def _lam_o(Sw, Sg, prm):
    return prm["k"] * kro(Sw, Sg, prm) / prm["mu_o"]


def _lam_g(Sg, prm):
    return prm["k"] * krg(Sg, prm) / prm["mu_g"]


# ------------------------------------------------------------------ 参数
def build_prm(case=0, fg_inj=0.5, N_f=10):
    import config
    c = config.case_params(case)
    return {
        "phi0": c["φ_0"], "cphi": c["c_φ"], "k": c["k_m0"],
        "rho_o": c["ρ_o"], "rho_w": c["ρ_w"],
        "mu_o": c["μ_o"], "mu_w": c["μ_w"],
        "co": c["c_o"], "cw": c["c_w"],
        "Sw0": c["S_w0"], "Swr": c["S_wr"], "Sor": c["S_or"],
        "kro0": c["k_ro0"], "krw0": c["k_rw0"],
        "no": c["n_o"], "nw": c["n_w"], "Ac": c["A_c"],
        "Lf": c["L_f"], "hf": c["h_f"], "qinj": c["q_inj"],
        "mu_co2": c["μ_CO2"],
        "p0": c["p_0"],
        # 气相参数：题面未给，取基准并做敏感性
        "rho_w_ref": 1000.0, "rho_g_ref": c["ρ_CO2,ref"],
        "rho_g": 850.0, "cg": 1.0 / c["p_0"], "mu_g": c["μ_CO2"],
        "krg0": 1.0, "ng": 2.0, "Sgc": 0.0,
        "fg_inj": fg_inj, "fw_inj": 1.0 - fg_inj,
        "p_max": c["p_max"],
        "p_wf": c["p_wf"],
        "N_f": N_f,
    }


class ThreePhase:
    def __init__(self, L=40.0, n=200, stretch=0.5, case=0, fg_inj=0.5,
                 p_end=None, q_inj_ref=None, N_f=10):
        self.prm = build_prm(case, fg_inj, N_f=N_f)
        if q_inj_ref is not None:
            self.prm["qinj"] = q_inj_ref
        self.prm["p_end"] = self.prm["p0"] if p_end is None else p_end

        self.xf, self.xc, self.dx = make_grid(L, n, stretch=stretch)
        self.N = len(self.dx)
        self.dxf = 0.5 * (self.dx[:-1] + self.dx[1:])

        self.P = np.full(self.N, self.prm["p0"])
        self.Sw = np.full(self.N, self.prm["Sw0"])
        self.Sg = np.zeros(self.N)
        self.t = 0.0

        area = self.prm["N_f"] * 4.0 * self.prm["Lf"] * self.prm["hf"]
        self.v_inj = self.prm["qinj"] / area   # config 已转 m3/s

        self.mass_in = np.zeros(3)
        self.mass_out = np.zeros(3)
        self.m0 = self._stored()

    # -------------------------------------------------------------- 状态
    def _stored(self):
        prm = self.prm
        ph = phi_of(self.P, prm)
        pw = self.P - pc(self.Sw, prm)
        So = np.clip(1.0 - self.Sw - self.Sg, 0.0, 1.0)
        return np.array([
            np.sum(self.dx * ph * rho_w_of(pw, prm) * self.Sw),
            np.sum(self.dx * ph * rho_o_of(self.P, prm) * So),
            np.sum(self.dx * ph * rho_g_of(self.P, prm) * self.Sg),
        ])

    def pack(self):
        u = np.empty(3 * self.N)
        u[0::3] = self.P
        u[1::3] = self.Sw
        u[2::3] = self.Sg
        return u

    # -------------------------------------------------------------- 单步残差
    def _cell_res(self, i, P, Sw, Sg, Pold, Swold, Sgold, dt_s):
        prm = self.prm
        dx = self.dx[i]
        ph = phi_of(P[i], prm)
        So = max(1.0 - Sw[i] - Sg[i], 1e-8)
        pw = P[i] - pc(Sw[i], prm)
        Aw = ph * rho_w_of(pw, prm) * Sw[i]
        Ao = ph * rho_o_of(P[i], prm) * So
        Ag = ph * rho_g_of(P[i], prm) * Sg[i]

        ph0 = phi_of(Pold[i], prm)
        So0 = max(1.0 - Swold[i] - Sgold[i], 1e-8)
        pw0 = Pold[i] - pc(Swold[i], prm)
        Aw0 = ph0 * rho_w_of(pw0, prm) * Swold[i]
        Ao0 = ph0 * rho_o_of(Pold[i], prm) * So0
        Ag0 = ph0 * rho_g_of(Pold[i], prm) * Sgold[i]

        if i == 0:
            FwL, FoL, FgL = self._injection_flux()
        else:
            FwL, FoL, FgL = self._face(i - 1, i, P, Sw, Sg)

        if i == self.N - 1:
            FwR, FoR, FgR = self._right_boundary(i, P, Sw, Sg)
        else:
            FwR, FoR, FgR = self._face(i, i + 1, P, Sw, Sg)

        Rw = dx * (Aw - Aw0) / dt_s + FwR - FwL
        Ro = dx * (Ao - Ao0) / dt_s + FoR - FoL
        Rg = dx * (Ag - Ag0) / dt_s + FgR - FgL
        return Rw, Ro, Rg

    def _face(self, i, j, P, Sw, Sg):
        prm = self.prm
        dxf = self.dxf[min(i, j)]
        pwl = P[i] - pc(Sw[i], prm)
        pwr = P[j] - pc(Sw[j], prm)
        if pwl >= pwr:
            lamw, rw = _lam_w(Sw[i], prm), rho_w_of(pwl, prm)
        else:
            lamw, rw = _lam_w(Sw[j], prm), rho_w_of(pwr, prm)
        Fw = -rw * lamw * (pwr - pwl) / dxf

        if P[i] >= P[j]:
            lamo, ro = _lam_o(Sw[i], Sg[i], prm), rho_o_of(P[i], prm)
            lamg, rg = _lam_g(Sg[i], prm), rho_g_of(P[i], prm)
        else:
            lamo, ro = _lam_o(Sw[j], Sg[j], prm), rho_o_of(P[j], prm)
            lamg, rg = _lam_g(Sg[j], prm), rho_g_of(P[j], prm)
        Fo = -ro * lamo * (P[j] - P[i]) / dxf
        Fg = -rg * lamg * (P[j] - P[i]) / dxf
        return Fw, Fo, Fg

    def _right_boundary(self, i, P, Sw, Sg):
        prm = self.prm
        p_bc = prm["p_end"]
        dxf = 0.5 * self.dx[i]
        if P[i] >= p_bc:
            Swb, Sgb = Sw[i], Sg[i]
        else:
            Swb, Sgb = prm["Sw0"], 0.0
        pwb = p_bc - pc(Swb, prm)
        Fw = -rho_w_of(pwb, prm) * _lam_w(Swb, prm) * (pwb - (P[i] - pc(Sw[i], prm))) / dxf
        Fo = -rho_o_of(p_bc, prm) * _lam_o(Swb, Sgb, prm) * (p_bc - P[i]) / dxf
        Fg = -rho_g_of(p_bc, prm) * _lam_g(Sgb, prm) * (p_bc - P[i]) / dxf
        return Fw, Fo, Fg

    def _injection_flux(self):
        prm = self.prm
        return (prm["rho_w_ref"] * self.v_inj * prm["fw_inj"], 0.0,
                prm["rho_g_ref"] * self.v_inj * prm["fg_inj"])

    def injection_pressure(self):
        """Estimate fracture-face oil pressure with frozen first-cell mobilities."""
        prm = self.prm
        p, sw, sg = self.P[0], self.Sw[0], self.Sg[0]
        water_in, _, gas_in = self._injection_flux()
        volume_flux = water_in / rho_w_of(p - pc(sw, prm), prm) + gas_in / rho_g_of(p, prm)
        mobility = _lam_w(sw, prm) + _lam_o(sw, sg, prm) + _lam_g(sg, prm)
        return p + volume_flux * (0.5 * self.dx[0]) / mobility

    def residual(self, u, uold, dt_s):
        P, Sw, Sg = u[0::3], u[1::3], u[2::3]
        Pold, Swold, Sgold = uold[0::3], uold[1::3], uold[2::3]
        R = np.empty_like(u)
        for i in range(self.N):
            Rw, Ro, Rg = self._cell_res(i, P, Sw, Sg, Pold, Swold, Sgold, dt_s)
            R[3 * i] = Rw
            R[3 * i + 1] = Ro
            R[3 * i + 2] = Rg
        return R

    def jacobian(self, u, uold, dt_s):
        N = self.N
        eps_p = 1e-6 * np.maximum(np.abs(u[0::3]), 1e5)
        eps_s = 1e-7
        rows, cols, vals = [], [], []

        base_R = np.empty((N, 3))
        for i in range(N):
            base_R[i] = self._cell_res(i, u[0::3], u[1::3], u[2::3],
                                       uold[0::3], uold[1::3], uold[2::3], dt_s)

        for i in range(N):
            lo, hi = max(0, i - 1), min(N - 1, i + 1)
            for j in range(lo, hi + 1):
                for var in (0, 1, 2):
                    k = 3 * j + var
                    eps = eps_p[j] if var == 0 else eps_s
                    u[k] += eps
                    Rp = np.array(self._cell_res(i, u[0::3], u[1::3], u[2::3],
                                                 uold[0::3], uold[1::3], uold[2::3], dt_s))
                    u[k] -= eps
                    dR = (Rp - base_R[i]) / eps
                    rows += [3 * i, 3 * i + 1, 3 * i + 2]
                    cols += [k, k, k]
                    vals += [dR[0], dR[1], dR[2]]

        return coo_matrix((vals, (rows, cols)), shape=(3 * N, 3 * N)).tocsr()

    # -------------------------------------------------------------- Newton
    def step(self, dt_s, maxit=25, tol=1e-10):
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("Time step must be positive and finite")
        u = self.pack()
        uold = u.copy()
        Pold, Swold, Sgold = uold[0::3].copy(), uold[1::3].copy(), uold[2::3].copy()

        scale = np.empty_like(u)
        for i in range(self.N):
            s = self.dx[i] * 1000.0 / dt_s
            scale[3 * i:3 * i + 3] = s

        R = self.residual(u, uold, dt_s)
        err = np.max(np.abs(R / scale))
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
                un[1::3] = np.clip(un[1::3], 1e-6, 1.0 - 1e-6)
                un[2::3] = np.clip(un[2::3], 0.0, 1.0 - 1e-6)
                So = 1.0 - un[1::3] - un[2::3]
                bad = So < 1e-6
                if np.any(bad):
                    un[2::3][bad] = np.maximum(0.0, 1.0 - un[1::3][bad] - 1e-6)
                un[0::3] = np.clip(un[0::3], 1e5, 1.2e8)
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

        self.P, self.Sw, self.Sg = u[0::3], u[1::3], u[2::3]
        self.t += dt_s / 86400.0

        self.mass_in += np.array(self._injection_flux()) * dt_s
        self.mass_out += np.array(
            self._right_boundary(self.N - 1, self.P, self.Sw, self.Sg)
        ) * dt_s
        return it, err

    def run(self, t_end, dt, maxit=25, tol=1e-10, verbose=False):
        nstep = int(round(t_end / dt))
        hist = []
        for k in range(nstep):
            it, err = advance(self, dt * 86400.0, maxit=maxit, tol=tol)
            if verbose and k % max(1, nstep // 10) == 0:
                print(f"  step {k}/{nstep}, it={it}, err={err:.2e}")
            hist.append((self.t, self.P.copy(), self.Sw.copy(), self.Sg.copy()))
        return hist

    def mass_balance(self):
        m = self._stored()
        res = self.m0 + self.mass_in - self.mass_out - m
        return {"initial": self.m0, "in": self.mass_in,
                "out": self.mass_out, "stored": m, "residual": res}
