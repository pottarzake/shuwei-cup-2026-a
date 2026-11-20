# -*- coding: utf-8 -*-
"""T11：纳米输运与 Langmuir 吸附验证。

1) Case3、N_f=10、q=300、10 d 的浓度剖面与质量守恒；
2) K_d 敏感性：0.01 / 0.005 / 0.002 / 0.001，看 0.01 wt% 半径是否可达。
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

from model.nano_transport import ThreePhaseNano
from model.timestepping import advance

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


def run_unit_sensitivity():
    rows = ["Case3, N_f=10, q=300 m3/d, n=50, injection only, 10 d.",
            "L/kg interpretation sensitivity. Radius is the farthest qualifying cell center."]
    for kd_l_per_kg in (0.02, 0.01, 0.005):
        model = make(kd_l_per_kg * 1e-3)
        while 10.0 - model.t > 1e-12:
            dt_s = min(0.02, 10.0 - model.t) * 86400.0
            advance(model, dt_s, tol=1e-9)
        balance = model.nano.mass_balance()
        rows.append(f"Kd={kd_l_per_kg:g} L/kg: C_first_cell={model.nano.C[0]:.3e}, "
                    f"radius={radius(model, C_TH):.2f} m, "
                    f"ads_frac={balance['adsorbed']/balance['in']:.3f}, "
                    f"rel_res={abs(balance['residual'])/balance['in']:.2e}")
    report = "\n".join(rows) + "\n"
    (OUT / "T11_Kd_L_per_kg_sensitivity.txt").write_text(report, encoding="utf-8")
    print(report, end="")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kd-l-per-kg", action="store_true", help="run the unit sensitivity only")
if parser.parse_args().kd_l_per_kg:
    run_unit_sensitivity()
    sys.exit(0)


# ---------------------------------------------------------------- 10 d 主算例
m = make()
snaps = {}
for target in (1.0, 5.0, 10.0):
    while target - m.t > 1e-12:
        advance(m, min(0.02, target - m.t) * 86400.0, tol=1e-9)
    snaps[target] = m.nano.C.copy()
    mb = m.nano.mass_balance()
    lines.append(
        f"t={target:.0f} d: C_first_cell={m.nano.C[0]:.3e}, C_max={m.nano.C.max():.3e}, "
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
        advance(mm, min(0.01, 1.0 - mm.t) * 86400.0, tol=1e-9)
    mb = mm.nano.mass_balance()
    lines.append(
        f"  K_d={kd:g}: C_first_cell={mm.nano.C[0]:.2e}, "
        f"r(0.01wt%)={radius(mm, C_TH):.2f} m, r(1e-5)={radius(mm, 1e-5):.2f} m, "
        f"ads_frac={mb['adsorbed']/max(mb['in'],1e-30):.3f}"
    )

lines.append("")
lines.append("适用范围：半径按达到阈值的最远单元中心报告。以上为 Case3 注入期算例，"
             "焖井期、其他情景和未采样参数仍待验证。")

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
