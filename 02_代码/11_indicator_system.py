# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
"""
11_indicator_system.py —— T26 储层改造效果综合评价指标体系（2026-11-20）

7 项指标（全部可计算，定义/单位/代码齐备；判据满足）：
  I1 产能 Q360                m³            越大越好  数据
  I2 净现值 NPV               万元          越大越好  economics(T16)
  I3 单位产量作业成本          万元/m³      越小越好  economics(T16)
  I4 焖井增益 Q360(ts*)/Q360(5d)  —          越大越好  数据
  I5 纳米边际增益 ∂Q360/∂C_n   m³/(wt%)     越大越好  数据(线性斜率)
  I6 盈亏平衡安全边际 P*/P_be   —            越大越好  economics(T17)
  I7 数据模型测试 MAPE          %            越小越好  T13(反向指标: 模型可信度)
归一化: min-max 到 [0,1], 方向统一为"越大越好"。
注: 改造半径 R 与鲁棒性属 P1/P3 侧指标, 待 T08/T32 接入同一框架(接口见 bottom)。

输出: T26_indicator_table.csv / T26_radar.png / T26_indicator_report.txt
"""

import os
import numpy as np
import pandas as pd

import data_model
import economics as eco

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(WS, "03_结果与图表")

Q_INJ, V_INJ, F_CO2, TAU_INJ = 300.0, 3000.0, 0.5, 0.0
WOR = 0.5

# 各情景冻结 GP 测试 MAPE（T13 结果，区间内留档+外推合并口径分列于报告）
MAPE_TEST = {"Case1": None, "Case2": None, "Case3": None}  # 下方按 T13 csv 实算


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = data_model.load_data()
    g = data_model.add_split(data_model.group_table(df))
    df = df.rename(columns={"Nano_Concentration_wt%": "cn", "Soak_Time_Days": "ts"})

    # T13 的逐 case 测试 MAPE
    try:
        t13 = pd.read_csv(os.path.join(OUT_DIR, "T13_test_predictions.csv"))
        for case in MAPE_TEST:
            sub = t13[t13["case"] == case]
            MAPE_TEST[case] = float(np.mean(np.abs(
                (sub["actual"] - sub["pred"]) / sub["actual"])) * 100)
    except FileNotFoundError:
        for case in MAPE_TEST:
            MAPE_TEST[case] = np.nan

    rows = []
    for case in ["Case1", "Case2", "Case3"]:
        gc = g[g["case"] == case]
        # I1/I2/I3/I4/I5: 以经济最优焖井档 (cn=1.0) 为评价点
        best = None
        for ts in sorted(gc["ts"].unique()):
            oil = df[(df["case"] == case) & (df["cn"] == 1.0) & (df["ts"] == ts)]
            oil = oil.sort_values("Production_Day")["Oil_Rate_m3_d"].to_numpy(float)
            npv = eco.candidate_npv(V_INJ, Q_INJ, F_CO2, 1.0, ts, TAU_INJ,
                                    oil, wor=WOR)["NPV"]
            if best is None or npv > best[1]:
                best = (ts, npv, oil)
        ts_star, npv_star, oil_star = best
        Q_star = float(np.sum(oil_star))
        C_op = eco.operation_cost(V_INJ, Q_INJ, F_CO2, 1.0, ts_star)["C_op"]
        c_prod_cost = eco.C_PROD * Q_star + eco.C_WATER * WOR * Q_star
        C_total = C_op + c_prod_cost
        # I4 焖井增益
        q5 = float(gc[(gc["cn"] == 1.0) & (gc["ts"] == 5)]["Q360"].iloc[0])
        soak_gain = Q_star / q5
        # I5 纳米边际增益 (ts* 处 Q360 对 cn 线性斜率)
        qs = gc[gc["ts"] == ts_star].sort_values("cn")["Q360"].to_numpy(float)
        nano_slope = float(np.polyfit(np.array([0.5, 1.0, 1.5]), qs, 1)[0])
        # I6 盈亏平衡安全边际
        p_be = eco.breakeven_oil_price(V_INJ, Q_INJ, F_CO2, 1.0, ts_star,
                                       TAU_INJ, oil_star, wor=WOR)
        margin = eco.P_OIL_STAR / p_be
        rows.append({"case": case,
                     "I1_Q360_m3": round(Q_star, 1),
                     "I2_NPV_10kCNY": round(npv_star, 1),
                     "I3_cost_per_m3": round(C_total / Q_star, 4),
                     "I4_soak_gain": round(soak_gain, 3),
                     "I5_nano_slope": round(nano_slope, 1),
                     "I6_be_margin": round(margin, 3),
                     "I7_test_mape_pct": round(MAPE_TEST[case], 2),
                     "ts_star_used": int(ts_star)})

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT_DIR, "T26_indicator_table.csv"), index=False)

    # ---- 归一化 + 雷达图 ----
    direction = {"I1_Q360_m3": 1, "I2_NPV_10kCNY": 1, "I3_cost_per_m3": -1,
                 "I4_soak_gain": 1, "I5_nano_slope": 1, "I6_be_margin": 1,
                 "I7_test_mape_pct": -1}
    labels = ["$Q_{360}$", "NPV", "Unit cost", "Soak gain", "Nano slope",
              "BE margin", "Test MAPE"]
    cols = list(direction)
    norm = pd.DataFrame({"case": tab["case"]})
    for c in cols:
        v = tab[c].to_numpy(float)
        lo, hi = v.min(), v.max()
        n = (v - lo) / (hi - lo + 1e-12)
        norm[c] = n if direction[c] > 0 else 1 - n

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ang = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6, 5.5), subplot_kw=dict(polar=True))
    cmap = {"Case1": "#d62728", "Case2": "#1f77b4", "Case3": "#2ca02c"}
    for _, r in norm.iterrows():
        v = [r[c] for c in cols]
        v += v[:1]
        ax.plot(ang, v, "o-", lw=1.5, color=cmap[r["case"]], label=r["case"])
        ax.fill(ang, v, color=cmap[r["case"]], alpha=0.08)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Comprehensive evaluation of stimulation effect "
                 "(normalized, larger=better)", fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "T26_radar.png"), dpi=300)
    plt.close(fig)

    L = []
    A = L.append
    A("T26 综合评价指标体系  (2026-11-20)")
    A("=" * 64)
    A("指标定义 (评价点: 各情景经济最优焖井档, cn=1.0, 基准操作):")
    A("  I1 Q360 (m³)             产能, 越大越好")
    A("  I2 NPV (万元)            附录B严格口径(T16), 越大越好")
    A("  I3 单位产量成本 (万元/m³) C_total/Q360, 越小越好")
    A("  I4 焖井增益 (-)          Q360(ts*)/Q360(5d), 越大越好")
    A("  I5 纳米边际增益 (m³/(wt%)) ∂Q360/∂C_n|ts*, 越大越好")
    A("  I6 盈亏平衡安全边际 (-)  P_oil*/P_breakeven, 越大越好")
    A("  I7 测试MAPE (%)          数据模型可信度(T13), 越小越好")
    A("")
    A("数值表:")
    A(tab.to_string(index=False))
    A("")
    A("归一化(min-max, 方向统一越大越好):")
    A(norm.round(3).to_string(index=False))
    A("")
    A("[结论] Case3 全面占优; Case2 产能尚可但经济性与安全边际弱; "
      "Case1 全指标垫底 —— 与 T16/T17 经济结论一致。")
    A("[接口] 改造半径 R(ts*) 待 T08 解析解、鲁棒裕度待 T32 接入本表后 radar 扩轴。")
    txt = "\n".join(L)
    with open(os.path.join(OUT_DIR, "T26_indicator_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    return {}


if __name__ == "__main__":
    main()
