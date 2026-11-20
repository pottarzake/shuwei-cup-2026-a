# -*- coding: utf-8 -*-
"""
T22 后续：从附件 C 反演 (case, Cn, ts) 维度的降阶生产参数。

对每个 (case, Cn)：
  使用首日产油和 360 天累计产油反演指数递减的初始瞬时油率及储集体积；
  L_d = V_o / (A * phi * movable)；
  分别拟合初始瞬时油率、L_d(ts) 为 a + b(1 - exp(-ts/c))。

再用拟合参数回算 Q360，检查整条曲线的误差。
这是数据驱动标定，不是机理预测；P3 用它承载焖井收益趋势。
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))
from model.production import fit_exponential_targets

XLSX = BASE / "01_数据" / "附件C_Synthetic_Production_Data.xlsx"
OUT = BASE / "03_结果与图表"

CASES = {1: (80000.0, 0.08, 0.58), 2: (96000.0, 0.10, 0.55), 3: (112000.0, 0.13, 0.52)}


def sat(t, a, b, c):
    return a + b * (1.0 - np.exp(-t / c))


def invert(group):
    rows = []
    for ts in sorted(group.Soak_Time_Days.unique()):
        g = group[group.Soak_Time_Days == ts]
        q1 = float(g[g.Production_Day == 1].Oil_Rate_m3_d.iloc[0])
        q360 = float(g[g.Production_Day == 360].Cum_Oil_m3.iloc[0])
        rate, inventory = fit_exponential_targets(q1, q360)
        rows.append((ts, rate, q360, inventory, q1))
    return rows


TRAIN_SOAK_DAYS = (5, 10, 20, 25, 30, 35, 40, 50)
all_data = pd.read_excel(XLSX)
df = all_data[all_data.Soak_Time_Days.isin(TRAIN_SOAK_DAYS)]
tab, detail = [], []
for case, (area, phi, movable) in CASES.items():
    for cn in (0.5, 1.0, 1.5):
        sub = df[(df.Case_ID.str.startswith(f"Case{case}")) &
                 (df["Nano_Concentration_wt%"] == cn)]
        rows = invert(sub)
        ts = np.array([r[0] for r in rows], float)
        q1 = np.array([r[1] for r in rows], float)
        q360 = np.array([r[2] for r in rows], float)
        Ld = np.array([r[3] for r in rows]) / (area * phi * movable)

        pL, _ = curve_fit(sat, ts, Ld, p0=[0.4, 0.1, 10], maxfev=20000)
        pq, _ = curve_fit(sat, ts, q1, p0=[20, 30, 10], maxfev=20000)
        Ld_pred = sat(ts, *pL)
        qi_pred = sat(ts, *pq)
        V_pred = area * phi * movable * Ld_pred
        q360_pred = -V_pred * np.expm1(-360.0 * qi_pred / V_pred)
        first_day_pred = -V_pred * np.expm1(-qi_pred / V_pred)
        first_day = np.array([r[4] for r in rows])
        rel = np.abs(q360_pred - q360) / q360

        tab.append([case, cn, *pL, *pq,
                    float(np.sqrt(np.mean((Ld - Ld_pred) ** 2))),
                    float(np.sqrt(np.mean((first_day - first_day_pred) ** 2))),
                    float(np.max(rel) * 100), float(np.mean(rel) * 100),
                    sat(30, *pL), sat(55, *pL)])
        for t, ld, ldp, qq, qp, a, b in zip(ts, Ld, Ld_pred, first_day, first_day_pred,
                                            q360, q360_pred):
            detail.append([case, cn, int(t), ld, ldp, qq, qp, a, b,
                           (b - a) / a * 100])

cols = ["case", "Cn", "Ld_a", "Ld_b", "Ld_c", "q_a", "q_b", "q_c",
        "Ld_rmse", "q_rmse", "Q360_maxerr_pct", "Q360_meanerr_pct",
        "Ld_30", "Ld_55"]
tab = pd.DataFrame(tab, columns=cols)
tab.to_csv(OUT / "T22_Ld_parameters.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(detail, columns=["case", "Cn", "ts", "Ld_inv", "Ld_fit",
                              "Q1_data", "Q1_fit", "Q360_data", "Q360_fit",
                              "Q360_err_pct"]).to_csv(
    OUT / "T22_Ld_detail.csv", index=False, encoding="utf-8-sig")

lines = ["降阶生产参数（仅训练档数据标定）",
         "qi(ts)=a+b(1-exp(-ts/c))；L_d(ts)=a+b(1-exp(-ts/c))",
         tab.round(4).to_string(index=False),
         "",
         "Q360 回算误差：训练档最大相对误差见 Q360_maxerr_pct。",
         "",
         "55 d 验证档，冻结系数并使用附件 C 的基准操作。以下为累计产油相对误差。"]
validation = all_data[(all_data.Soak_Time_Days == 55) & (all_data.Production_Day == 360)]
for case, (area, phi, movable) in CASES.items():
    errors = []
    for row in tab[tab.case == case].itertuples():
        observed = validation.loc[
            validation.Case_ID.str.startswith(f"Case{case}_")
            & validation["Nano_Concentration_wt%"].eq(row.Cn), "Cum_Oil_m3"
        ].item()
        inventory = area * phi * movable * row.Ld_55
        rate = sat(55, row.q_a, row.q_b, row.q_c)
        prediction = -inventory * np.expm1(-360 * rate / inventory)
        errors.append(abs(prediction - observed) / observed)
    lines.append(f"Case{case}: mean={np.mean(errors):.2%}, max={max(errors):.2%}")
(OUT / "T22_Ld_parameters.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
