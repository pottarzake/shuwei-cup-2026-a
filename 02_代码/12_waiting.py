# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
"""T23：等待期 τ_inj 对缝宽、导流与 NPV 的影响，并分离折现与闭合两条路径。"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

from model.p3_forward import evaluate
from model.waiting import state

OUT = BASE / "03_结果与图表"

# T29 零指数主跑的折中候选
CAND = {
    2: dict(cn=0.50, q=136.0, V=1000.0, f=0.79, ts=21.0),
    3: dict(cn=0.51, q=800.0, V=1000.0, f=0.80, ts=21.0),
}
TAUS = np.arange(0, 16, 1.0)

rows = []
for case, p in CAND.items():
    for tau in TAUS:
        s = state(tau, case)
        total = evaluate(case, p["cn"], p["q"], p["V"], p["f"], p["ts"], tau,
                         frac_mult=s["mult"])
        delay = evaluate(case, p["cn"], p["q"], p["V"], p["f"], p["ts"], tau,
                         frac_mult=1.0)
        cond = evaluate(case, p["cn"], p["q"], p["V"], p["f"], p["ts"], 0.0,
                        frac_mult=s["mult"])
        rows.append([case, tau, s["p_frac"] / 1e6, s["sigma_n"] / 1e6,
                     s["w"] * 1e3, s["mult"], total["NPV"], delay["NPV"],
                     cond["NPV"]])

df = pd.DataFrame(rows, columns=["case", "tau", "p_frac_MPa", "sigma_n_MPa",
                                 "w_mm", "k_mult", "NPV_total", "NPV_delay",
                                 "NPV_cond"])
df.to_csv(OUT / "T23_waiting.csv", index=False, encoding="utf-8-sig")

lines = ["T23 等待期：缝宽闭合与 NPV 影响",
         "p_f(tau)=p0+(p_f-p0)e^{-tau/tau_c}, tau_c=2 d",
         "w=w_inf+(w0-w_inf)e^{-c_c(sigma_n-sigma_n0)}, c_c=0.05 MPa^-1, w_inf=0.2w0",
         "导流折减 = (w/w0)^3"]
for case in (2, 3):
    sub = df[df.case == case]
    r0, r15 = sub.iloc[0], sub.iloc[-1]
    lines.append(f"Case{case}: tau=0 -> w={r0.w_mm:.2f} mm, NPV={r0.NPV_total:.0f}; "
                 f"tau=15 -> w={r15.w_mm:.2f} mm, k_mult={r15.k_mult:.3f}, "
                 f"NPV_total={r15.NPV_total:.0f} (delay-only {r15.NPV_delay:.0f}, "
                 f"cond-only {r15.NPV_cond:.0f})")
lines.append("")
lines.append("tau_c、c_c、w_inf 为假设值，不是附件 C 标定结果；结论只说明趋势与量级。")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, case in zip(axes, (2, 3)):
    sub = df[df.case == case]
    ax.plot(sub.tau, sub.NPV_total, "o-", label="total")
    ax.plot(sub.tau, sub.NPV_delay, "s--", label="delay only")
    ax.plot(sub.tau, sub.NPV_cond, "^:", label="closure only")
    ax.set_title(f"Case{case}")
    ax.set_xlabel("waiting time (d)"); ax.set_ylabel("NPV (10^4 CNY)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "T23_waiting.png", dpi=200)
plt.close(fig)

(OUT / "T23_waiting.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
