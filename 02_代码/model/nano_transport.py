# -*- coding: utf-8 -*-
"""
T11：纳米颗粒在水相中的输运 + Langmuir 吸附（算子分裂）。

在每个已接受的三相流动步之后，用该步的压力/饱和度场求解标量输运：

    d/dt[phi S_w rho_w C + rho_r (1-phi) C_a]
    + d/dx[rho_w C v_w] - d/dx[phi S_w rho_w D_n dC/dx] = 0

    C_a = C_a,max K_L C_w / (1 + K_L C_w),  C_w = rho_w C,  K_L = K_d / C_a,max

低浓度极限 C_a ~ K_d C_w，与题给线性系数一致；高浓度自动饱和。
"""

import numpy as np
from scipy.linalg import solve_banded

from .three_phase import ThreePhase, pc, phi_of, rho_w_of, _lam_w


class NanoTransport:
    def __init__(self, xc, dx, dxf, prm, case_params, beta_s=0.5, mode='langmuir'):
        self.xc = xc
        self.dx = dx
        self.dxf = dxf
        self.N = len(dx)
        self.prm = prm

        self.C_inj = case_params["C_n_wt"] / 100.0          # 质量分数
        self.Dn = case_params["D_n"]
        self.Kd = case_params["K_d"]
        self.rho_r = case_params["ρ_r"]

        phi0 = prm["phi0"]
        rho_w0 = prm["rho_w"]
        # 饱和水条件下的可用纳米量 / 干岩质量
        self.available = (self.C_inj * phi0 * rho_w0
                          / ((1.0 - phi0) * self.rho_r))
        self.Ca_max = beta_s * self.available
        self.KL = self.Kd / self.Ca_max                     # m3/kg
        self.mode = mode

        self.C = np.zeros(self.N)
        self.mass_in = 0.0
        self.mass_out = 0.0
        self.m0 = 0.0  # 初始 C=0，水相与吸附量均为 0
        self._phi_old = None

    # -------------------------------------------------------------- 吸附
    def Ca_of(self, C, rho_w):
        Cw = rho_w * C
        if self.mode == "linear":
            return self.Kd * Cw
        return self.Ca_max * self.KL * Cw / (1.0 + self.KL * Cw)

    def dCa_dC(self, C, rho_w):
        Cw = rho_w * C
        if self.mode == "linear":
            return self.Kd * rho_w
        return self.Ca_max * self.KL * rho_w / (1.0 + self.KL * Cw) ** 2

    def stored(self):
        phi = np.array([phi_of(p, self.prm) for p in self._P])
        rho_w = np.array([rho_w_of(p - pc(s, self.prm), self.prm)
                          for p, s in zip(self._P, self._Sw)])
        water = np.sum(self.dx * phi * self._Sw * rho_w * self.C)
        ads = np.sum(self.dx * self.rho_r * (1.0 - phi) *
                     self.Ca_of(self.C, rho_w))
        return water, ads

    # -------------------------------------------------------------- 速度
    def _velocities(self, model):
        prm = self.prm
        P, Sw, Sg = model.P, model.Sw, model.Sg
        pw = P - np.array([pc(s, prm) for s in Sw])
        rho_w = np.array([rho_w_of(p, prm) for p in pw])

        vf = np.zeros(self.N - 1)
        for f in range(self.N - 1):
            if pw[f] >= pw[f + 1]:
                lam, rw = _lam_w(Sw[f], prm), rho_w[f]
            else:
                lam, rw = _lam_w(Sw[f + 1], prm), rho_w[f + 1]
            vf[f] = -lam * (pw[f + 1] - pw[f]) / self.dxf[f]

        v_left = model.v_inj * prm["fw_inj"]
        p_end = prm["p_end"]
        if P[-1] >= p_end:
            pw_bc = p_end - pc(Sw[-1], prm)
            lam = _lam_w(Sw[-1], prm)
            v_right = -lam * (pw_bc - pw[-1]) / (0.5 * self.dx[-1])
        else:
            v_right = 0.0
        return vf, v_left, v_right, pw, rho_w

    # -------------------------------------------------------------- 单步
    def step(self, dt_s, model, maxit=20, tol=1e-10):
        self._P = model.P
        self._Sw = model.Sw
        prm = self.prm
        phi = np.array([phi_of(p, prm) for p in model.P])
        Sw = model.Sw
        vf, v_left, v_right, pw, rho_w = self._velocities(model)

        # 面扩散系数
        Df = np.zeros(self.N - 1)
        for f in range(self.N - 1):
            phif = 0.5 * (phi[f] + phi[f + 1])
            swf = 0.5 * (Sw[f] + Sw[f + 1])
            rwf = 0.5 * (rho_w[f] + rho_w[f + 1])
            Df[f] = phif * swf * rwf * self.Dn / self.dxf[f]

        C = self.C.copy()
        Cold = C.copy()
        rho_r = self.rho_r

        def accum(Ci, ph, Si, rwi):
            return (ph * Si * rwi * Ci
                    + rho_r * (1.0 - ph) * self.Ca_of(Ci, rwi))

        # 旧的孔隙度/饱和度/密度必须用上一步的值，否则算子分裂会破坏守恒
        if self._phi_old is None:
            self._phi_old = phi.copy()
            self._Sw_old = Sw.copy()
            self._rho_old = rho_w.copy()
        A0 = np.array([accum(Cold[i], self._phi_old[i], self._Sw_old[i],
                             self._rho_old[i]) for i in range(self.N)])

        def face_flux(f, Cl, Cr):
            """面 f 的纳米通量，返回 (F, dF/dCl, dF/dCr)。"""
            v = vf[f]
            if v >= 0:
                F = rho_w[f] * Cl * v
                dFl, dFr = rho_w[f] * v, 0.0
            else:
                F = rho_w[f + 1] * Cr * v
                dFl, dFr = 0.0, rho_w[f + 1] * v
            F -= Df[f] * (Cr - Cl)
            return F, dFl + Df[f], dFr - Df[f]

        for it in range(1, maxit + 1):
            rows = np.zeros(self.N)
            lower = np.zeros(self.N)
            upper = np.zeros(self.N)
            rhs = np.zeros(self.N)

            for i in range(self.N):
                Ci = C[i]
                Ai = accum(Ci, phi[i], Sw[i], rho_w[i])
                dA = (phi[i] * Sw[i] * rho_w[i]
                      + rho_r * (1.0 - phi[i]) * self.dCa_dC(Ci, rho_w[i]))
                r = self.dx[i] * (Ai - A0[i]) / dt_s

                # 左面
                if i == 0:
                    FL = rho_w[0] * v_left * self.C_inj
                    dFL_dCi = 0.0
                else:
                    FL, dFL_dCl, dFL_dCi = face_flux(i - 1, C[i - 1], Ci)
                # 右面
                if i == self.N - 1:
                    if v_right >= 0:
                        FR = rho_w[-1] * Ci * v_right
                        dFR_dCi = rho_w[-1] * v_right
                    else:
                        FR, dFR_dCi = 0.0, 0.0
                else:
                    FR, dFR_dCi, dFR_dCr = face_flux(i, Ci, C[i + 1])

                rows[i] = self.dx[i] * dA / dt_s + dFR_dCi - dFL_dCi
                if i > 0:
                    lower[i] = -dFL_dCl
                if i < self.N - 1:
                    upper[i] = dFR_dCr
                rhs[i] = -(r + FR - FL)

            ab = np.zeros((3, self.N))
            ab[0, 1:] = upper[:-1]
            ab[1, :] = rows
            ab[2, :-1] = lower[1:]
            dC = solve_banded((1, 1), ab, rhs)
            C = np.clip(C + dC, 0.0, 0.2)

            if np.max(np.abs(dC)) < tol * max(1.0, np.max(C)):
                break

        self.C = C
        self._phi_old = phi.copy()
        self._Sw_old = Sw.copy()
        self._rho_old = rho_w.copy()
        self.mass_in += rho_w[0] * v_left * self.C_inj * dt_s
        if v_right >= 0:
            self.mass_out += rho_w[-1] * C[-1] * v_right * dt_s
        return it

    def mass_balance(self):
        water, ads = self.stored()
        return {
            "in": self.mass_in, "out": self.mass_out,
            "water": float(water), "adsorbed": float(ads),
            "initial": self.m0,
            "residual": self.m0 + self.mass_in - self.mass_out - water - ads,
        }


class ThreePhaseNano(ThreePhase):
    """三相流动 + 纳米输运的算子分裂模型。"""

    def __init__(self, *args, beta_s=0.5, case=0, nano_mode='langmuir', **kwargs):
        super().__init__(*args, case=case, **kwargs)
        import config
        self.nano = NanoTransport(self.xc, self.dx, self.dxf, self.prm,
                                  config.case_params(case),
                                  beta_s=beta_s, mode=nano_mode)

    def step(self, dt_s, **kwargs):
        it, err = super().step(dt_s, **kwargs)
        self.nano.step(dt_s, self)
        return it, err

    def mass_balance(self):
        mb = super().mass_balance()
        mb["nano"] = self.nano.mass_balance()
        return mb
