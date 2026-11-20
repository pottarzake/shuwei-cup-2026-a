# -*- coding: utf-8 -*-
"""T08：压力再分布解析解与数值焖井的对照（轻量版）。

n=100，注入 dt=0.1 d，焖井 dt=0.2 d。每个快照点写进度，便于监控。
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
from model.three_phase import ThreePhase
from model.timestepping import advance, ConvergenceError
from model.p2_scaling import fit_step_profile, redistribution, radius

OUT = BASE / "03_结果与图表"
OUT.mkdir(exist_ok=True)
PROG = OUT / "T08_progress.txt"

CASE, NF, Q = 3, 10, 300.0
L, N = 40.0, 100
THR_FRAC = 0.05

lines = [f"Case{CASE}, N_f={NF}, q={Q:g}, L={L:g}, n={N}"]
PROG.write_text("start\n", encoding="utf-8")

m = ThreePhase(L=L, n=N, stretch=0.5, case=CASE, fg_inj=0.5,
               q_inj_ref=Q / 86400.0, N_f=NF)
p0 = m.prm["p0"]
thr = THR_FRAC * p0
lines.append(f"p0={p0/1e6:.2f} MPa, threshold={thr/1e6:.3f} MPa")

# ---------------------------------------------------------------- 注入 10 d
try:
    while 10.0 - m.t > 1e-12:
        advance(m, min(0.1, 10.0 - m.t) * 86400.0, tol=1e-9, max_halvings=4)
    PROG.write_text(f"injection done at {m.t:.2f} d\n", encoding="utf-8")
except ConvergenceError as e:
    lines.append(f"injection convergence failure at t={m.t:.3f} d: {e}")
    PROG.write_text(f"injection FAIL at {m.t:.2f}\n", encoding="utf-8")

dp0 = m.P - p0
A, a = fit_step_profile(m.xc, dp0)
D = config.hydraulic_diffusivity(CASE)
lines.append(f"injection end: t={m.t:.2f} d, p_face={m.P[0]/1e6:.2f} MPa, "
             f"A={A/1e6:.3f} MPa, a={a:.2f} m, D={D:.3e} m2/s")

# ---------------------------------------------------------------- 关井
m.v_inj = 0.0
t0 = m.t                       # 注入结束时刻，焖井时长从这里起算
targets = [1, 2, 5, 10, 20, 30, 45, 60]
rows, snaps = [], {}
for ts in targets:
    target_t = t0 + ts
    try:
        while target_t - m.t > 1e-12:
            advance(m, min(0.2, target_t - m.t) * 86400.0, tol=1e-9, max_halvings=4)
    except ConvergenceError as e:
        msg = f"soak FAIL at t={m.t:.2f} d: {e}"
        lines.append(msg)
        with PROG.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
        break

    dp = m.P - p0
    idx = np.where(dp <= thr)[0]
    if len(idx) == 0:
        R_num = float(m.xc[-1])
    elif idx[0] == 0:
        R_num = 0.0 if dp[0] <= thr else float(m.xc[0])
    else:
        R_num = float(m.xc[idx[0]])
    R_ana = radius(ts, D, A, a, p0, THR_FRAC)
    rows.append((ts, R_num, R_ana))
    snaps[ts] = (m.xc.copy(), dp.copy())
    msg = (f"ts={ts:>4.1f} d: R_num={R_num:6.2f}, R_ana={R_ana:6.2f}, "
           f"dp_face={dp[0]/1e6:.3f} MPa")
    lines.append(msg)
    with PROG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")

lines.append("")
lines.append("域长检查：4*sqrt(D*ts) 接近 L 时半无限假设失效。")
for ts in (30, 60):
    lines.append(f"  ts={ts}: 4*sqrt(D*ts)={4*np.sqrt(D*ts*86400):.1f} m (L={L:g})")

# ---------------------------------------------------------------- 图
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(m.xc, dp0 / 1e6, "k-", lw=2, label="t=0 (inj end)")
for ts, (x, dp) in snaps.items():
    if ts in (1, 5, 20, 60):
        axes[0].plot(x, dp / 1e6, label=f"num {ts}d")
        axes[0].plot(x, redistribution(x, ts, D, A, a) / 1e6, "--", lw=1,
                     label=f"ana {ts}d")
axes[0].axhline(thr / 1e6, color="gray", ls=":", lw=1)
axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("p - p0 (MPa)")
axes[0].set_xlim(0, 40); axes[0].legend(fontsize=6, ncol=2); axes[0].grid(alpha=.3)

if rows:
    ts_arr = np.array([r[0] for r in rows])
    rn = np.array([r[1] for r in rows])
    ra = np.array([r[2] for r in rows])
    axes[1].loglog(ts_arr, np.maximum(rn, 1e-2), "o-", label="numerical")
    axes[1].loglog(ts_arr, np.maximum(ra, 1e-2), "s--", label="analytical")
axes[1].set_xlabel("soak time (d)"); axes[1].set_ylabel("R (m)")
axes[1].legend(fontsize=8); axes[1].grid(alpha=.3, which="both")

fig.tight_layout(); fig.savefig(OUT / "T08_p2_redistribution.png", dpi=200)
plt.close(fig)

(OUT / "T08_p2_redistribution_check.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
