# -*- coding: utf-8 -*-
"""P3：三情景 NSGA-II 多目标优化 + 鲁棒性。

目标：max Q_o，min C_total。约束：p_max（q<=q_max）、纳米<=30 t、
注入+焖井<=90 d。V/f 的缩放指数是待验证假设。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

from model.p3_forward import evaluate, decode, NAMES, CASE
from model.nsga2 import run

OUT = BASE / "03_结果与图表"
OUT.mkdir(exist_ok=True)
SEED = 20261120


def make_eval(case):
    def f(x):
        cn, q, V, fc, ts, tau = decode(x)
        r = evaluate(case, cn, q, V, fc, ts, tau)
        return -r["Q_o"], r["C_total"], r["violation"]
    return f


lines = ["P3 NSGA-II（pop=100, gen=80）",
         "目标 max Q_o / min C_total；约束 p_max、纳米<=30 t、注入+焖井<=90 d"]
fronts = {}

for case in (1, 2, 3):
    X, F, CV, cache = run(make_eval(case), pop=100, gen=80, seed=SEED + case)
    ok = CV <= 1e-12
    pts = []
    for x, cv in zip(X[ok], CV[ok]):
        cn, q, V, fc, ts, tau = decode(x)
        r = evaluate(case, cn, q, V, fc, ts, tau)
        pts.append([cn, q, V, fc, ts, tau, r["Q_o"], r["C_total"], r["NPV"],
                    r["t_inj"], r["m_nano"], r["q1"], r["Ld"]])
    df = pd.DataFrame(pts, columns=["cn", "q", "V", "f", "ts", "tau",
                                    "Q_o", "C_total", "NPV", "t_inj",
                                    "m_nano", "q1", "Ld"])
    if not df.empty:
        df = df.sort_values("Q_o").drop_duplicates(subset=["Q_o"]).reset_index(drop=True)
        df.to_csv(OUT / f"T29_pareto_case{case}.csv", index=False, encoding="utf-8-sig")
        fronts[case] = df
        lines.append(f"Case{case}: 可行解 {len(df)} 个；Q_o {df.Q_o.min():.0f}–"
                     f"{df.Q_o.max():.0f} m3，C_total {df.C_total.min():.0f}–"
                     f"{df.C_total.max():.0f} 万元")
    else:
        i = int(np.argmin(CV))
        cn, q, V, fc, ts, tau = decode(X[i])
        lines.append(f"Case{case}: 无可行解；最小约束违反 {CV[i]:.3f}，"
                     f"对应 q={q:.0f} (q_max={CASE[case][3]:.0f})")

lines.append("")
lines.append("折中方案（膝点 + Q_o 的 25/75 分位）：")
robust_rows = []
for case, df in fronts.items():
    if df.empty:
        continue
    qn = (df.Q_o - df.Q_o.min()) / max(df.Q_o.max() - df.Q_o.min(), 1e-9)
    cn_ = (df.C_total - df.C_total.min()) / max(df.C_total.max() - df.C_total.min(), 1e-9)
    dist = np.sqrt((1 - qn) ** 2 + cn_ ** 2)
    picks = [int(np.argmin(dist)),
             int(np.argmin(np.abs(qn - 0.25))),
             int(np.argmin(np.abs(qn - 0.75)))]
    picks = sorted(set(picks))
    for k, i in enumerate(picks, 1):
        r = df.iloc[i]
        lines.append(f"  Case{case}-S{k}: cn={r.cn:.2f}, q={r.q:.0f}, V={r.V:.0f}, "
                     f"f={r.f:.2f}, ts={r.ts:.0f}, tau={r.tau:.1f}; "
                     f"Q_o={r.Q_o:.0f}, C={r.C_total:.0f}, NPV={r.NPV:.0f}")
        # 鲁棒性
        vals, infeas = [], 0
        rng = np.random.default_rng(SEED + 10 * case + k)
        for _ in range(200):
            kr, pr, sr = rng.uniform(0.85, 1.15, 3)
            rr = evaluate(case, r.cn, r.q, r.V, r.f, r.ts, r.tau,
                          k_r=kr, phi_r=pr, sh_r=sr)
            if rr["feasible"]:
                vals.append(rr["NPV"])
            else:
                infeas += 1
        vals = np.array(vals)
        robust_rows.append([case, k, r.cn, r.q, r.V, r.f, r.ts, r.tau,
                            r.Q_o, r.C_total, r.NPV,
                            len(vals) / 200.0,
                            float(np.mean(vals)) if len(vals) else np.nan,
                            float(np.std(vals)) if len(vals) else np.nan,
                            float(np.mean(np.sort(vals)[:max(1, len(vals)//10)])) if len(vals) else np.nan,
                            float(np.mean(vals > 0)) if len(vals) else np.nan])

rob = pd.DataFrame(robust_rows, columns=["case", "scheme", "cn", "q", "V", "f",
                                         "ts", "tau", "Q_o", "C_total", "NPV",
                                         "feasible_rate", "NPV_mean", "NPV_std",
                                         "NPV_worst10", "prob_NPV_pos"])
rob.to_csv(OUT / "T30_robust.csv", index=False, encoding="utf-8-sig")
if not rob.empty:
    lines.append("")
    lines.append("鲁棒性（±15% k/phi/sigma_h，200 次）：")
    for _, r in rob.iterrows():
        lines.append(f"  Case{int(r.case)}-S{int(r.scheme)}: 可行率 {r.feasible_rate:.2f}, "
                     f"P(NPV>0)={r.prob_NPV_pos:.2f}, worst10={r.NPV_worst10:.0f}, "
                     f"mean±sd={r.NPV_mean:.0f}±{r.NPV_std:.0f}")

# ---------------------------------------------------------------- 图
fig, ax = plt.subplots(figsize=(6, 4))
for case, df in fronts.items():
    ax.plot(df.C_total, df.Q_o, "o-", ms=3, label=f"Case{case}")
ax.set_xlabel("C_total (10^4 CNY)"); ax.set_ylabel("Q_o (m3)")
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "T29_pareto.png", dpi=200); plt.close(fig)

(OUT / "T29_pareto_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
