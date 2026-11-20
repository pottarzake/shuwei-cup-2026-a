# -*- coding: utf-8 -*-
"""T09：两相渗流验证——质量守恒、压力峰值、饱和度前缘。"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

from model.two_phase import TwoPhase, pc

OUT = BASE / "03_结果与图表"
OUT.mkdir(exist_ok=True)

m = TwoPhase(L=40.0, n=200, stretch=0.5, case=0, fw_inj=0.5)
prm = m.prm

lines = [
    f"Case0, L=40 m, n={m.N}, fw_inj={prm['fw_inj']}",
    f"v_inj = {m.v_inj:.4e} m/s, q_inj = {prm['qinj'] * 86400:.1f} m3/d",
    f"p_max = {prm['p_max'] / 1e6:.1f} MPa, p0 = {prm['p0'] / 1e6:.1f} MPa",
]

# 注入能力估算：稳态 q = lambda_t * A * dp / L
from model.two_phase import krw, kro
area = 4.0 * prm["Lf"] * prm["hf"]
dp = prm["p_max"] - prm["p0"]
lam_t = (prm["k"] * krw(prm["Sw0"], prm) / prm["mu_w"]
         + prm["k"] * kro(prm["Sw0"], prm) / prm["mu_o"])
lam_co2 = prm["k"] / prm["mu_co2"]
q_feas = lam_t * area * dp / 40.0 * 86400
q_co2 = lam_co2 * area * dp / 40.0 * 86400
lines.append(f"total mobility at Sw0 = {lam_t:.3e} m2/(Pa s)")
lines.append(f"feasible q at dp=12 MPa (oil-water) = {q_feas:.2f} m3/d")
lines.append(f"same with CO2 mobility only       = {q_co2:.1f} m3/d")
lines.append("NOTE: 油水两相不含低黏度 CO2 相，注入能力被低估；"
             "P3 的压力约束会大量绑定，Q2 需重新评估自由相。")

snaps = {}
t_targets = [0.01, 0.05, 0.1, 0.2]
dt = 1e-3
t_done = 0.0
max_it = 0
for tt in t_targets:
    nstep = int(round((tt - t_done) / dt))
    if nstep <= 0:
        continue
    for _ in range(nstep):
        it, err = m.step(dt * 86400.0, tol=1e-11)
        max_it = max(max_it, it)
        if not np.isfinite(err):
            raise RuntimeError("Newton diverged")
    t_done = tt
    snaps[tt] = (m.P.copy(), m.S.copy())
    lines.append(f"t={tt:.2f} d: p0={m.P[0]/1e6:.3f} MPa, "
                 f"peak p={m.P.max()/1e6:.3f} MPa, "
                 f"Sw0={m.S[0]:.5f}, Sw_max={m.S.max():.5f}")

mb = m.mass_balance()
lines.append("mass balance:")
for k, v in mb.items():
    lines.append(f"  {k}: {v:.6e}")
w_rel = abs(mb["water_residual"]) / max(abs(mb["water_in"]), 1e-30)
o_rel = abs(mb["oil_residual"]) / max(abs(mb["oil_in"]), 1e-30)
lines.append(f"water rel residual = {w_rel:.3e}")
lines.append(f"oil   rel residual = {o_rel:.3e}")
lines.append(f"max Newton iterations = {max_it}")
lines.append(f"peak p <= p_max: {'PASS' if m.P.max() <= prm['p_max'] else 'FAIL'}")
lines.append(f"mass balance < 1e-6: {'PASS' if max(w_rel, o_rel) < 1e-6 else 'FAIL'}")

# 剖面图
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for tt, (P, S) in snaps.items():
    axes[0].plot(m.xc, (P - prm["p0"]) / 1e6, label=f"t={tt} d")
    axes[1].plot(m.xc, S, label=f"t={tt} d")
axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("p - p0 (MPa)")
axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("S_w")
for ax in axes:
    ax.grid(alpha=.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "T09_two_phase_profiles.png", dpi=200)
plt.close(fig)

(OUT / "T09_two_phase_check.txt").write_text("\n".join(lines), encoding="utf-8")
print("done")
