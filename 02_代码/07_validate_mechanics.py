# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
"""T18/T19：一维孔隙弹性应变与压缩损伤量级校验。

结论用于判断：题给损伤阈值（0.002–0.010）在三情景的实际压力变化下
是否会触发损伤，以及需要多小的阈值才会触发。
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
from model.mechanics import (constrained_modulus, uniaxial_strain,
                             damage_from_strain_history)

OUT = BASE / "03_结果与图表"
OUT.mkdir(exist_ok=True)

lines = ["T18/T19 一维孔隙弹性与压缩损伤",
         "应变拉伸为正；注入 dp>0 -> eps>0（不产生压缩损伤）；生产 dp<0 -> eps<0"]
rows = []
for case in (1, 2, 3):
    c = config.case_params(case)
    M = constrained_modulus(c["E"], c["ν"])
    e_inj = uniaxial_strain(c["p_max"], c["p_0"], c["α"], M)
    e_prod = uniaxial_strain(c["p_wf"], c["p_0"], c["α"], M)
    kappa = max(-e_inj, -e_prod, 0.0)
    d = 1.0 - np.exp(-c["β_d"] * max(kappa - c["ε_d"], 0.0))
    rows.append((case, M, e_inj, e_prod, kappa, c["ε_d"], d))
    lines.append(
        f"Case{case}: E={c['E']/1e9:.1f} GPa, nu={c['ν']}, alpha={c['α']}, "
        f"M={M/1e9:.1f} GPa, eps_inj={e_inj:+.2e}, eps_prod={e_prod:+.2e}, "
        f"kappa={kappa:.2e}, eps_d={c['ε_d']:.3f}, d={d:.3e}"
    )

lines.append("")
lines.append("临界阈值（使 d 首次 >0）：eps_d < kappa")
for case, M, e_inj, e_prod, kappa, eps_d, d in rows:
    lines.append(f"  Case{case}: eps_d_crit = {kappa:.2e}")

lines.append("")
lines.append("阈值敏感性（生产压差下的 d）：")
thresholds = [1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
lines.append("  eps_d     " + "  ".join(f"Case{r[0]}" for r in rows))
for ed in thresholds:
    vals = []
    for case, M, e_inj, e_prod, kappa, _, _ in rows:
        c = config.case_params(case)
        vals.append(1.0 - np.exp(-c["β_d"] * max(kappa - ed, 0.0)))
    lines.append(f"  {ed:.0e}  " + "  ".join(f"{v:.3e}" for v in vals))

lines.append("")
lines.append("结论：题给 eps_d=0.004–0.006 远大于实际压缩应变 2–4e-4，"
             "三情景名义损伤 d=0。损伤要在 eps_d 降到 ~3e-4 以下才会触发；"
             "论文应如实报告，并把该阈值作为敏感性参数。")

# ---------------------------------------------------------------- 图
fig, ax = plt.subplots(figsize=(6, 4))
for case, M, e_inj, e_prod, kappa, _, _ in rows:
    c = config.case_params(case)
    ed = np.logspace(-5, -2, 200)
    d = 1.0 - np.exp(-c["β_d"] * np.maximum(kappa - ed, 0.0))
    ax.semilogx(ed, d, label=f"Case{case} (kappa={kappa:.1e})")
ax.axvspan(0.004, 0.006, color="gray", alpha=.2, label="given eps_d range")
ax.set_xlabel("damage threshold eps_d"); ax.set_ylabel("d")
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "T18_damage_sensitivity.png", dpi=200)
plt.close(fig)

(OUT / "T18_mechanics_check.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
