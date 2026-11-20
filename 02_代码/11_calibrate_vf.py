# -*- coding: utf-8 -*-
"""
T31：标定 P3 的 V、f 缩放指数。

Case3、N_f=10、q=300、Cn=1、ts=30。跑 (V,f) 网格的三相注入+焖井，
用焖井末态按降阶模型算 Q360，再拟合：

    ln(Q/Q0) = a ln(V/V0) + b ln(f/f0)

V0=3000, f0=0.5。注入全程检查 p_max；超压的点标为不可行。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

from model.three_phase import (ThreePhase, kro, krw, pc, phi_of)
from model.timestepping import advance, ConvergenceError

OUT = BASE / "03_结果与图表"
CASE, NF, Q, L, N = 3, 10, 300.0, 40.0, 100
T_SOAK = 30.0
LD_30 = 0.457          # T22: Case3/Cn=1/ts=30
AREA, PHI0, SOR = 112000.0, 0.13, 0.20
P_WF = 35e6

GRID = [(1000, 0.2), (1000, 0.8), (3000, 0.2), (3000, 0.5),
        (3000, 0.8), (5000, 0.2), (5000, 0.8)]


def tank_q(m):
    prm = m.prm
    p_avg = float(np.mean(m.P))
    so = np.clip(1 - m.Sw - m.Sg, 0, 1)
    movable = float(np.mean(np.clip(so - SOR, 0, 1)))
    phi = float(np.mean(phi_of(m.P, prm)))
    Vo = AREA * LD_30 * phi * movable
    lam_o = float(np.mean(prm["k"] * kro(m.Sw, m.Sg, prm) / prm["mu_o"]))
    lam_w = float(np.mean(prm["k"] * krw(m.Sw, prm) / prm["mu_w"]))
    dp = max(p_avg - prm["p_wf"], 0.0)
    qi = lam_o * dp / L * AREA * 86400.0
    qw = lam_w * dp / L * AREA * 86400.0
    if qi + qw > 100.0:
        s = 100.0 / (qi + qw)
        qi *= s; qw *= s
    tau = Vo / qi if qi > 0 else np.inf
    Q360 = Vo * (1 - np.exp(-360.0 / tau)) if np.isfinite(tau) else 0.0
    return Q360, qi, tau, p_avg, movable


lines = ["T31 V/f 指数标定：Case3, N_f=10, q=300, Cn=1, ts=30",
         "V(m3), f, t_inj(d), 状态, Q360(m3), q_i(m3/d), tau(d), p_avg(MPa), movable"]
rows = []
for V, f in GRID:
    m = ThreePhase(L=L, n=N, stretch=0.5, case=CASE, fg_inj=f,
                   q_inj_ref=Q / 86400.0, N_f=NF)
    t_inj = V / Q
    cap = m.prm["p_max"]
    status = "ok"
    try:
        while t_inj - m.t > 1e-12:
            advance(m, min(0.1, t_inj - m.t) * 86400.0, tol=1e-9,
                    max_halvings=4,
                    on_step=lambda: m.injection_pressure() <= cap)
            if m.t < t_inj - 1e-9 and m.injection_pressure() > cap:
                status = f"p_max at {m.t:.2f} d"
                break
        if status == "ok":
            m.v_inj = 0.0
            end = m.t + T_SOAK
            while end - m.t > 1e-12:
                advance(m, min(0.2, end - m.t) * 86400.0, tol=1e-9,
                        max_halvings=4)
    except ConvergenceError as e:
        status = f"conv fail at {m.t:.2f}: {e}"

    if status == "ok":
        Q360, qi, tau, p_avg, movable = tank_q(m)
    else:
        Q360 = qi = tau = p_avg = movable = np.nan
    rows.append([V, f, t_inj, status, Q360, qi, tau, p_avg, movable])
    lines.append(f"{V:5.0f}, {f:3.1f}, {t_inj:5.2f}, {status}, "
                 f"{Q360:.1f}, {qi:.2f}, {tau:.1f}, {p_avg/1e6:.2f}, {movable:.3f}")

df = pd.DataFrame(rows, columns=["V", "f", "t_inj", "status", "Q360",
                                 "qi", "tau", "p_avg", "movable"])
df.to_csv(OUT / "T31_vf_calibration.csv", index=False, encoding="utf-8-sig")

# 对数拟合（只用 ok 点，含基准点）
ok = df.dropna(subset=["Q360"])
base = ok[(ok.V == 3000) & (np.isclose(ok.f, 0.5))]
if len(base) and len(ok) >= 3:
    Q0 = float(base.Q360.iloc[0])
    A = np.column_stack([np.log(ok.V / 3000.0), np.log(ok.f / 0.5)])
    y = np.log(ok.Q360 / Q0)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    lines.append("")
    lines.append(f"拟合：a(V 指数)={coef[0]:.3f}, b(f 指数)={coef[1]:.3f}, R2={r2:.3f}, n={len(ok)}")
    lines.append(f"基准 Q0(V=3000,f=0.5)={Q0:.1f} m3")
else:
    lines.append("有效点不足，无法拟合。")

(OUT / "T31_vf_calibration.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
