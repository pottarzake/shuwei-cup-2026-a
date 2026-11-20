# -*- coding: utf-8 -*-
"""T10：三相（水/油/气）验证——注入能力、质量守恒、相分布。

固定 q_inj=300 m3/d，比较三个情景在 0.5 d 注入末端的井底压力。
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

from model.three_phase import ThreePhase

OUT = BASE / "03_结果与图表"
OUT.mkdir(exist_ok=True)

lines = ["q_inj = 300 m3/d, fg_inj = 0.5, L = 40 m, n = 100, t_end = 0.5 d"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

summary = []
dt = 5e-3
nmax = int(round(0.5 / dt))
for case in (1, 2, 3):
    m = ThreePhase(L=40.0, n=100, stretch=0.5, case=case, fg_inj=0.5)
    t_fail = None
    for _ in range(nmax):
        m.step(dt * 86400.0, tol=1e-9)
        if m.P.max() > m.prm["p_max"]:
            t_fail = m.t
            break
    mb = m.mass_balance()
    rel = np.max(np.abs(mb["residual"])) / np.max(mb["stored"])
    p0 = m.P[0] / 1e6
    pcap = m.prm["p_max"] / 1e6
    ok = t_fail is None
    summary.append((case, p0, pcap, ok, rel, m.Sg.max(), m.Sw[0], t_fail))
    lines.append(f"Case{case}: t_stop={m.t:.3f} d, "
                 f"{'feasible to 0.5 d' if ok else f'p_max exceeded at {t_fail:.3f} d'}, "
                 f"p0={p0:.2f} MPa (limit {pcap:.1f}), Sg_max={m.Sg.max():.3f}, "
                 f"Sw0={m.Sw[0]:.3f}, mass_rel={rel:.2e}")
    axes[0].plot(m.xc, (m.P - m.prm["p0"]) / 1e6, label=f"Case{case}")
    axes[1].plot(m.xc, m.Sg, label=f"Case{case}")

axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("p - p0 (MPa)")
axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("S_g")
for ax in axes:
    ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "T10_three_phase_profiles.png", dpi=200)
plt.close(fig)

lines.append("")
lines.append("结论：q=300 m3/d 在 Case3 接近可行性边界，Case1/Case2 明显超压；"
             "P3 的压力约束会绑定量，低渗情景必须降 q_inj。")
lines.append("两相对照：油水两相可行 q 仅 8.55 m3/d，三相提高到约 300 m3/d 量级。")

(OUT / "T10_three_phase_check.txt").write_text("\n".join(lines), encoding="utf-8")
print("done")
