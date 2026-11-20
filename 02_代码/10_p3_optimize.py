# -*- coding: utf-8 -*-
"""P3：三情景 NSGA-II 多目标优化 + 鲁棒性。

目标：max Q_o，min C_total。约束：p_max（q<=q_max）、纳米<=30 t、
注入+焖井<=90 d。V/f 的缩放指数是待验证假设。
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

from model.p3_forward import evaluate, decode, CASE
from model.nsga2 import fast_nds, run
from model.three_phase import ThreePhase
from model.timestepping import ConvergenceError, advance

OUT = BASE / "03_结果与图表"
OUT.mkdir(exist_ok=True)
SEED = 20261120


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--positive-vf", action="store_true",
                    help="rerun the exploratory 0.2/0.1 exponent assumption")
args = parser.parse_args()
exponents = (0.2, 0.1) if args.positive_vf else (0.0, 0.0)
filename_suffix = "_positive_vf" if args.positive_vf else ""


def output_file(name, extension):
    return OUT / f"{name}{filename_suffix}.{extension}"


def evaluate_scheme(*args, **kwargs):
    return evaluate(*args, exp_v=exponents[0], exp_f=exponents[1], **kwargs)


def make_eval(case):
    def f(x):
        cn, q, V, fc, ts, tau = decode(x)
        r = evaluate_scheme(case, cn, q, V, fc, ts, tau)
        return -r["Q_o"], r["C_total"], r["violation"]
    return f


lines = ["P3 NSGA-II（pop=100, gen=80）",
         "目标 max Q_o / min C_total。粗筛使用固定流度估算的 q_max。",
         "附件 C 仅训练档标定。τ_inj=0；等待期闭合效应待接入。",
         f"V/f 指数设定为 {exponents[0]:g}/{exponents[1]:g}，产水用 WOR=0.5。"
         "NPV 为绝对值，B1 待校准。"]
if args.positive_vf:
    lines.append("正指数为假设压力测试，三相压力复核仅在零指数主跑执行。")
else:
    lines.append("零指数为保守建模约定，T31 仅检验 Case3 的局部降阶响应。")
fronts = {}

for case in (1, 2, 3):
    X, F, CV, _ = run(make_eval(case), n_var=5, pop=100, gen=80, seed=SEED + case)
    first_front, _ = fast_nds(F, CV)
    candidates = [i for i in first_front[0] if CV[i] <= 1e-12]
    pts = []
    for i in candidates:
        x = X[i]
        cn, q, V, fc, ts, tau = decode(x)
        r = evaluate_scheme(case, cn, q, V, fc, ts, tau)
        pts.append([cn, q, V, fc, ts, tau, r["Q_o"], r["C_total"], r["NPV"],
                    r["t_inj"], r["m_nano"], r["q1"], r["Ld"]])
    df = pd.DataFrame(pts, columns=["cn", "q", "V", "f", "ts", "tau",
                                    "Q_o", "C_total", "NPV", "t_inj",
                                    "m_nano", "q1", "Ld"])
    df = (df.sort_values(["Q_o", "C_total"])
          .drop_duplicates(subset=["Q_o", "C_total"]).reset_index(drop=True))
    df.to_csv(output_file(f"T29_pareto_case{case}", "csv"), index=False, encoding="utf-8-sig")
    if not df.empty:
        fronts[case] = df
        lines.append(f"Case{case}: 粗筛非支配点 {len(df)} 个；Q_o {df.Q_o.min():.0f}–"
                     f"{df.Q_o.max():.0f} m3，C_total {df.C_total.min():.0f}–"
                     f"{df.C_total.max():.0f} 万元")
    else:
        i = int(np.argmin(CV))
        cn, q, V, fc, ts, tau = decode(X[i])
        lines.append(f"Case{case}: 粗筛无可行点。最小约束违反 {CV[i]:.3f}，"
                     f"对应 q={q:.0f} (q_max={CASE[case][3]:.0f})")

lines.append("")
lines.append("折中候选（归一化理想点距离 + 产量跨度 25%/75%）：")
def verify_pressure(case, row):
    """Recheck the nominee over the full injection duration."""
    model = ThreePhase(L=40.0, n=100, stretch=0.5, case=case, fg_inj=row.f,
                       q_inj_ref=row.q / 86400, N_f=10)
    peak = model.prm["p0"]
    limit = model.prm["p_max"]

    def record():
        nonlocal peak
        peak = max(peak, model.injection_pressure())
        return peak <= limit

    try:
        while row.t_inj - model.t > 1e-12:
            dt_s = min(0.05, row.t_inj - model.t) * 86400
            result = advance(model, dt_s, tol=1e-9, on_step=record)
            if result is None:
                return "overpressure", peak / 1e6, limit / 1e6, model.t
    except ConvergenceError:
        return "numerical_failure", peak / 1e6, limit / 1e6, model.t
    return "within_limit", peak / 1e6, limit / 1e6, model.t


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
    picks = set(picks)
    if len(picks) < min(2, len(df)):
        picks.add(int(np.argmax(np.abs(qn - qn.iloc[next(iter(picks))]))))
    picks = sorted(picks)
    samples = np.random.default_rng(SEED + 10 * case).uniform(0.85, 1.15, (200, 3))
    for k, i in enumerate(picks, 1):
        r = df.iloc[i]
        lines.append(f"  Case{case}-S{k}: cn={r.cn:.2f}, q={r.q:.0f}, V={r.V:.0f}, "
                     f"f={r.f:.2f}, ts={r.ts:.0f}, tau={r.tau:.1f}; "
                     f"Q_o={r.Q_o:.0f}, C={r.C_total:.0f}, NPV={r.NPV:.0f}")
        # 鲁棒性
        vals = []
        for kr, pr, sr in samples:
            rr = evaluate_scheme(case, r.cn, r.q, r.V, r.f, r.ts, r.tau,
                          k_r=kr, phi_r=pr, sh_r=sr)
            if rr["feasible"]:
                vals.append(rr["NPV"])
        vals = np.array(vals)
        if args.positive_vf:
            pressure_status, pressure_peak, pressure_limit, pressure_time = "not_run", np.nan, np.nan, np.nan
        else:
            pressure_status, pressure_peak, pressure_limit, pressure_time = verify_pressure(case, r)
        print(f"Case{case}-S{k}: pressure recheck {pressure_status}", flush=True)
        robust_rows.append([case, k, r.cn, r.q, r.V, r.f, r.ts, r.tau,
                            r.Q_o, r.C_total, r.NPV,
                            len(vals) / 200.0,
                            float(np.mean(vals)) if len(vals) else np.nan,
                            float(np.std(vals)) if len(vals) else np.nan,
                            float(np.mean(np.sort(vals)[:max(1, len(vals)//10)])) if len(vals) else np.nan,
                            float(np.mean(vals > 0)) if len(vals) else np.nan,
                            float(np.count_nonzero(vals > 0) / len(samples)),
                            pressure_status, pressure_peak, pressure_limit, pressure_time])

rob = pd.DataFrame(robust_rows, columns=["case", "scheme", "cn", "q", "V", "f",
                                         "ts", "tau", "Q_o", "C_total", "NPV",
                                         "feasible_rate", "NPV_mean", "NPV_std",
                                         "NPV_worst10_conditional", "prob_NPV_pos_conditional",
                                         "prob_feasible_and_NPV_pos", "pressure_status_3phase",
                                         "pressure_peak_MPa", "pressure_limit_MPa", "pressure_stop_d"])
rob.to_csv(output_file("T30_robust", "csv"), index=False, encoding="utf-8-sig")
if not rob.empty:
    lines.append("")
    lines.append("粗筛鲁棒性（±15% k/phi/sigma_h，200 次，情景内共用扰动样本）：")
    for _, r in rob.iterrows():
        lines.append(f"  Case{int(r.case)}-S{int(r.scheme)}: 可行率 {r.feasible_rate:.2f}, "
                     f"P(可行且NPV>0)={r.prob_feasible_and_NPV_pos:.2f}, "
                     f"条件P(NPV>0)={r.prob_NPV_pos_conditional:.2f}, "
                     f"条件worst10={r.NPV_worst10_conditional:.0f}, "
                     f"条件mean±sd={r.NPV_mean:.0f}±{r.NPV_std:.0f}")
        if r.pressure_status_3phase == "not_run":
            continue
        lines.append(f"    名义三相注入复核: {r.pressure_status_3phase}, "
                     f"估算峰值/上限 {r.pressure_peak_MPa:.2f}/{r.pressure_limit_MPa:.2f} MPa, "
                     f"停止时刻 {r.pressure_stop_d:.2f} d")

# ---------------------------------------------------------------- 图
fig, ax = plt.subplots(figsize=(6, 4))
for case, df in fronts.items():
    ax.plot(df.C_total, df.Q_o, "o-", ms=3, label=f"Case{case}")
ax.set_xlabel("C_total (10^4 CNY)"); ax.set_ylabel("Q_o (m3)")
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(output_file("T29_pareto", "png"), dpi=200); plt.close(fig)

output_file("T29_pareto_summary", "txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
