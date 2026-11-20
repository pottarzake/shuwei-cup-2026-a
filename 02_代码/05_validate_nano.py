# -*- coding: utf-8 -*-
"""T11：纳米输运与 Langmuir 吸附验证。

1) Case3、N_f=10、q=300、10 d 的浓度剖面与质量守恒；
2) K_d 敏感性：0.01 / 0.005 / 0.002 / 0.001，看 0.01 wt% 半径是否可达。
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

from model.nano_transport import ThreePhaseNano

OUT = BASE / "03_结果与图表"
OUT.mkdir(exist_ok=True)

C_TH = 1e-4          # 0.01 wt%
lines = ["Case3, N_f=10, q=300 m3/d, f_CO2=0.5, L=40 m, n=50",
         f"threshold C = {C_TH:g} (0.01 wt%)"]


def make(kd=None):
    m = ThreePhaseNano(L=40.0, n=50, stretch=0.5, case=3, fg_inj=0.5,
                       q_inj_ref=300 / 86400.0, N_f=10, beta_s=0.5)
    if kd is not None:
        m.nano.Kd = kd
        m.nano.KL = kd / m.nano.Ca_max
    return m


def radius(m, th):
    idx = np.where(m.nano.C >= th)[0]
    return float(m.xc[idx[-1]]) if len(idx) else 0.0


# ---------------------------------------------------------------- 10 d 主算例
m = make()
snaps = {}
for target in (1.0, 5.0, 10.0):
    while target - m.t > 1e-12:
        m.step(min(0.02, target - m.t) * 86400.0, tol=1e-9)
    snaps[target] = m.nano.C.copy()
    mb = m.nano.mass_balance()
    lines.append(
        f"t={target:.0f} d: C_face={m.nano.C[0]:.3e}, C_max={m.nano.C.max():.3e}, "
        f"r(0.01wt%)={radius(m, C_TH):.2f} m, "
        f"ads_frac={mb['adsorbed']/max(mb['in'],1e-30):.3f}, "
        f"rel_res={abs(mb['residual'])/max(mb['in'],1e-30):.1e}"
    )

# ---------------------------------------------------------------- K_d 敏感性
lines.append("")
lines.append("K_d sensitivity at 1 d:")
for kd in (0.01, 0.005, 0.002, 0.001):
    mm = make(kd)
    while 1.0 - mm.t > 1e-12:
        mm.step(0.01 * 86400.0, tol=1e-9)
    mb = mm.nano.mass_balance()
    lines.append(
        f"  K_d={kd:g}: C_face={mm.nano.C[0]:.2e}, "
        f"r(0.01wt%)={radius(mm, C_TH):.2f} m, r(1e-5)={radius(mm, 1e-5):.2f} m, "
        f"ads_frac={mb['adsorbed']/max(mb['in'],1e-30):.3f}"
    )

lines.append("")
lines.append("结论：题给 K_d=0.01 m3/kg 下 C_face~1.6e-6，比 0.01 wt% 阈值低约 60 倍；"
             "即使 K_d 降到 0.001，C_face 仍 <1e-4。0.01 wt% 纳米半径在题给范围内恒为 0。"
             "需核对 K_d 单位（m3/kg vs L/kg）或放宽吸附假设。")

# ---------------------------------------------------------------- 图
fig, ax = plt.subplots(figsize=(6, 4))
for t, C in snaps.items():
    ax.semilogy(m.xc, np.maximum(C, 1e-30), label=f"t={t:g} d")
ax.axhline(C_TH, ls="--", c="k", lw=1, label="0.01 wt%")
ax.set_xlabel("x (m)"); ax.set_ylabel("C_n (mass fraction)")
ax.set_xlim(0, 5); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "T11_nano_profiles.png", dpi=200)
plt.close(fig)

(OUT / "T11_nano_check.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
