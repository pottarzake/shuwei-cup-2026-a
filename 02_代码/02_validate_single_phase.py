# -*- coding: utf-8 -*-
"""T06/T07：单相扩散校验、网格与时间收敛、域长敏感性。

关键结论（见输出）：
- 空间为 2 阶，时间为一阶（向后 Euler）。
- 首层厚度必须随网格整体加密而缩小，否则空间阶会被时间误差掩盖。
- 生产网格建议 n=400、stretch≈0.5、注入/早期焖井 dt≈1e-3 d。
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

import config
from model.flow import solve_diffusion, erfc_solution

OUT = BASE / "03_结果与图表"
OUT.mkdir(exist_ok=True)

D = config.hydraulic_diffusivity(0)
p0, pL = 45e6, 57e6
L_CHECK = 200.0


def rel_err(p_num, p_ana):
    return np.linalg.norm(p_num - p_ana) / np.linalg.norm(p_ana - p0)


lines = [f"D = {D:.6e} m2/s"]
profile_errors = []

# ---------------------------------------------------------------- 1) erfc 对比
fig, ax = plt.subplots(figsize=(6, 4))
for t_d in (1, 5, 10):
    t = t_d * 86400.0
    xc, p, _ = solve_diffusion(L_CHECK, D, p0, pL, p0, t, 1e-3 * 86400,
                               n=400, first=0.01)
    pa = erfc_solution(xc, t, D, p0, pL)
    e = rel_err(p, pa)
    profile_errors.append(e)
    lines.append(f"t={t_d:>2} d: rel_L2_vs_erfc = {e:.3e}")
    ax.plot(xc, (p - p0) / 1e6, "o", ms=3, label=f"num t={t_d}d")
    ax.plot(xc, (pa - p0) / 1e6, "-", lw=1, label=f"erfc t={t_d}d")
ax.set_xlim(0, 60); ax.set_xlabel("x (m)"); ax.set_ylabel("p - p0 (MPa)")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "T06_erfc_profiles.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 2) 空间收敛
# 均匀网格，dt 按 dx^2 缩放，使时间误差低于空间误差
t = 5 * 86400.0
ns = [100, 200, 400, 800]
errs = []
for n in ns:
    dx = L_CHECK / n
    dt = 0.1 * dx * dx / D
    xc, p, _ = solve_diffusion(L_CHECK, D, p0, pL, p0, t, dt,
                               n=n, stretch=1.0)
    errs.append(rel_err(p, erfc_solution(xc, t, D, p0, pL)))
orders = [np.log(errs[i] / errs[i + 1]) / np.log(ns[i + 1] / ns[i])
          for i in range(len(ns) - 1)]
lines.append("grid: " + ", ".join(f"n={n}:{e:.2e}" for n, e in zip(ns, errs)))
lines.append("grid order: " + ", ".join(f"{o:.2f}" for o in orders))
grid_ok = all(1.6 <= o <= 2.4 for o in orders)

# ---------------------------------------------------------------- 3) 时间收敛
# 固定细网格与很小的首层，隔离空间误差
n_fine, first_fine = 400, 0.0025
dts = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4]
terrs = []
for dt in dts:
    xc, p, _ = solve_diffusion(L_CHECK, D, p0, pL, p0, t, dt * 86400,
                               n=n_fine, first=first_fine)
    terrs.append(rel_err(p, erfc_solution(xc, t, D, p0, pL)))
torders = [np.log(terrs[i] / terrs[i + 1]) / np.log(dts[i] / dts[i + 1])
           for i in range(len(dts) - 1)]
lines.append("time: " + ", ".join(f"dt={d:g}d:{e:.2e}" for d, e in zip(dts, terrs)))
lines.append("time order: " + ", ".join(f"{o:.2f}" for o in torders))
time_ok = all(0.7 <= o <= 1.3 for o in torders)

# ---------------------------------------------------------------- 4) 生产网格候选
# 注意：L=40 时 t>=5 d 压力已接近右边界，不能再用半无限 erfc 作参考。
# 这里用同域细网格解作参考，误差反映纯数值误差。
lines.append("production candidates (L=40 m, vs same-domain reference):")
ref = {}
for td in (1, 5, 10):
    tt = td * 86400.0
    xc_r, p_r, _ = solve_diffusion(40.0, D, p0, pL, p0, tt, 5e-4 * 86400,
                                   n=1600, stretch=1.0)
    ref[td] = (xc_r, p_r)

cands = [(400, 0.5, 1e-3), (400, 0.5, 5e-4), (200, 0.6, 2e-3)]
for n, st, dt in cands:
    row = []
    for td in (1, 5, 10):
        tt = td * 86400.0
        xc, p, _ = solve_diffusion(40.0, D, p0, pL, p0, tt, dt * 86400,
                                   n=n, stretch=st)
        pr = np.interp(xc, ref[td][0], ref[td][1])
        row.append(np.linalg.norm(p - pr) / np.linalg.norm(pr - p0))
    lines.append(f"  n={n} stretch={st} dt={dt:g}: "
                 f"{row[0]:.2e}/{row[1]:.2e}/{row[2]:.2e}")

# ---------------------------------------------------------------- 5) 域长敏感性
lines.append("domain sensitivity (t=30 d, n=400, stretch=0.5, dt=0.02 d):")
for Ld in (30.0, 40.0, 60.0):
    xc, p, _ = solve_diffusion(Ld, D, p0, pL, p0, 30 * 86400, 0.02 * 86400,
                               n=400, stretch=0.5)
    thr = 0.05 * p0
    x_with_faces = np.r_[0.0, xc, Ld]
    disturbance = np.r_[pL - p0, p - p0, 0.0]
    R = np.interp(thr, disturbance[::-1], x_with_faces[::-1])
    lines.append(f"  L={Ld:>4.0f} m: p(0)-p0={pL-p0:.3e} Pa, "
                 f"radius at 5%p0 = {R:.2f} m")

# ---------------------------------------------------------------- 图
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].loglog(ns, errs, "o-")
axes[0].set_xlabel("n"); axes[0].set_ylabel("rel L2 error")
axes[0].set_title("grid convergence (dt ~ dx^2)"); axes[0].grid(alpha=.3, which="both")
axes[1].loglog(dts, terrs, "s-")
axes[1].set_xlabel("dt (d)"); axes[1].set_ylabel("rel L2 error")
axes[1].set_title("time convergence"); axes[1].grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "T06_convergence.png", dpi=200); plt.close(fig)

profile_ok = all(error < 0.01 for error in profile_errors)
lines.append(f"CHECK erfc relative error < 1%: {'PASS' if profile_ok else 'FAIL'}")
lines.append(f"CHECK grid order in [1.6,2.4]: {'PASS' if grid_ok else 'FAIL'}")
lines.append(f"CHECK time order in [0.7,1.3]: {'PASS' if time_ok else 'FAIL'}")
(OUT / "T06_single_phase_check.txt").write_text("\n".join(lines), encoding="utf-8")
if not (profile_ok and grid_ok and time_ok):
    raise SystemExit("Single-phase validation failed. See T06_single_phase_check.txt.")
print("Single-phase validation passed.")
