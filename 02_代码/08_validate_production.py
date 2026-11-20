# -*- coding: utf-8 -*-
"""T21/T22：降阶生产模型标定（Case3，Cn=1，ts=30）。"""

import sys
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

from model.three_phase import ThreePhase
from model.timestepping import advance, ConvergenceError
from model.production import tank_production

OUT = BASE / "03_结果与图表"
OUT.mkdir(exist_ok=True)

CASE, NF, Q = 3, 10, 300.0
L, N = 40.0, 100
T_INJ = 10.0
T_SOAK = 30.0
Q_DATA = 3434.0     # 附件 C: Case3, Cn=1.0, ts=30, Q360

m = ThreePhase(L=L, n=N, stretch=0.5, case=CASE, fg_inj=0.5,
               q_inj_ref=Q / 86400.0, N_f=NF)
lines = [f"Case{CASE}, N_f={NF}, q={Q:g}, L={L:g}, n={N}, "
         f"t_inj={T_INJ:g}, t_soak={T_SOAK:g}"]
lines.append(f"data Q360 = {Q_DATA:.1f} m3 (Cn=1.0, ts=30)")

try:
    while T_INJ - m.t > 1e-12:
        advance(m, min(0.1, T_INJ - m.t) * 86400.0, tol=1e-9, max_halvings=4)
    lines.append(f"injection end: t={m.t:.2f}, p_face={m.P[0]/1e6:.2f} MPa")
    m.v_inj = 0.0
    t_soak_end = m.t + T_SOAK
    while t_soak_end - m.t > 1e-12:
        advance(m, min(0.2, t_soak_end - m.t) * 86400.0, tol=1e-9, max_halvings=4)
    lines.append(f"soak end: t={m.t:.2f}, p_avg={m.P.mean()/1e6:.2f} MPa, "
                 f"Sw_avg={m.Sw.mean():.3f}, Sg_avg={m.Sg.mean():.4f}")
except ConvergenceError as e:
    lines.append(f"FAIL at t={m.t:.2f}: {e}")

area = NF * 4.0 * m.prm["Lf"] * m.prm["hf"]
V_bulk = area * L
res = tank_production(m.P, m.Sw, m.Sg, m.prm, area, L, V_bulk,
                      t_end=360.0, liquid_cap=100.0, eta=1.0)
lines.append(f"area={area:.0f} m2, V_bulk={V_bulk:.3e} m3")
lines.append(f"q_i_o={res['q_i_o']:.3f} m3/d, q_i_w={res['q_i_w']:.3f} m3/d")
lines.append(f"V_o={res['V_o']:.3e} m3, tau={res['tau']:.1f} d")
lines.append(f"Q_o(eta=1)={res['Q_o']:.1f} m3")
eta = Q_DATA / res["Q_o"] if res["Q_o"] > 0 else np.nan
lines.append(f"calibration eta = {eta:.4f}")

# 用有效产油层厚 L_d 替代 40 m 全域体积，二分标定到数据
def q_for_Ld(Ld):
    return tank_production(m.P, m.Sw, m.Sg, m.prm, area, L, area * Ld,
                           t_end=360.0, liquid_cap=100.0, eta=1.0)["Q_o"]
lo, hi = 1e-3, L
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if q_for_Ld(mid) < Q_DATA:
        lo = mid
    else:
        hi = mid
Ld = 0.5 * (lo + hi)
res3 = tank_production(m.P, m.Sw, m.Sg, m.prm, area, L, area * Ld,
                       t_end=360.0, liquid_cap=100.0, eta=1.0)
lines.append(f"effective drained length L_d = {Ld:.3f} m (calibrated, eta=1)")
lines.append(f"Q_o(L_d)={res3['Q_o']:.1f} m3, q_i_o={res3['q_i_o']:.2f} m3/d, "
             f"tau={res3['tau']:.1f} d")
lines.append("")
lines.append("说明：绝对量级由单点标定；P3 只用相对趋势。"
             "其他情景的 eta 需分别标定，且 Case1/2 的基准注入在压力约束下不可行。")

(OUT / "T21_production_calibration.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
