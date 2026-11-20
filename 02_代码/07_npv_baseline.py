# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
# OpenAI Codex (GPT-6 family, exact release date unconfirmed) assisted review or edits on 2026-11-20.
"""
07_npv_baseline.py —— T16 NPV 模块运行 + T17 盈亏平衡与三类亏损口径（2026-11-20）

输入：附件 C 日度产量（Q_w 缺失，用 WOR 假设，默认 0.5，附敏感性）
基准操作参数（Case0）：q_inj=300 m³/d, V_inj=3000 m³, f_CO2=0.5, τ_inj=0
输出（../03_结果与图表/）：
  T16_npv_selfcheck.txt   848.25 自检 + 口径说明
  T16_npv_ts_curves.csv   三情景 × 三浓度 × 12 焖井档 NPV
  T16_npv_ts_curves.png   NPV–t_s 曲线（找最优焖井时间）
  T17_breakeven_report.txt 盈亏平衡油价 + 三类亏损口径表

依赖：economics.py、data_model.py（numpy/pandas/matplotlib）
"""

import os
import numpy as np
import pandas as pd

import data_model
import economics as eco

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(WS, "03_结果与图表")

# 基准操作参数（A.3 表 Case0 列；t_s 与 C_n 扫描）
Q_INJ, V_INJ, F_CO2, TAU_INJ = 300.0, 3000.0, 0.5, 0.0
WOR_DEFAULT = 0.5
CNS = [0.5, 1.0, 1.5]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ok, total, parts, detail = eco.selfcheck_848_25()
    assert ok, "848.25 自检失败"

    df = data_model.load_data()
    df = df.rename(columns={"Nano_Concentration_wt%": "cn", "Soak_Time_Days": "ts"})

    # ---- 扫描 NPV(t_s) ----
    rows = []
    for case in ["Case1", "Case2", "Case3"]:
        for cn in CNS:
            for ts in sorted(df["ts"].unique()):
                oil = df[(df["case"] == case) & (df["cn"] == cn) & (df["ts"] == ts)]
                oil = oil.sort_values("Production_Day")["Oil_Rate_m3_d"].to_numpy(float)
                r = eco.candidate_npv(V_INJ, Q_INJ, F_CO2, cn, ts, TAU_INJ,
                                      oil, wor=WOR_DEFAULT)
                rows.append({"case": case, "cn": cn, "ts": int(ts),
                             "NPV": round(r["NPV"], 2),
                             "Q_o": round(r["Q_o"], 1),
                             "nano_ok": r["nano_ok"]})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "T16_npv_ts_curves.csv"), index=False)

    # ---- 图：NPV–t_s ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    cols = {0.5: "#1f77b4", 1.0: "#ff7f0e", 1.5: "#2ca02c"}
    for ax, case in zip(axes, ["Case1", "Case2", "Case3"]):
        for cn in CNS:
            s = res[(res["case"] == case) & (res["cn"] == cn)].sort_values("ts")
            ax.plot(s["ts"], s["NPV"], "o-", ms=4, lw=1.2, color=cols[cn],
                    label=f"$C_n$={cn} wt%")
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_title(case)
        ax.set_xlabel("Soak time $t_s$ (d)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("NPV ($10^4$ CNY)")
    fig.suptitle(f"NPV vs soak time (baseline ops: $q_{{inj}}$=300 m$^3$/d, "
                 f"$V_{{inj}}$=3000 m$^3$, $f_{{CO2}}$=0.5, WOR={WOR_DEFAULT})",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, "T16_npv_ts_curves.png"), dpi=300)
    plt.close(fig)

    # ---- 最优焖井时间 + 盈亏平衡油价（绝对 NPV=0 口径）----
    opt_rows, be_rows = [], []
    for case in ["Case1", "Case2", "Case3"]:
        for cn in CNS:
            s = res[(res["case"] == case) & (res["cn"] == cn)]
            i = s["NPV"].idxmax()
            ts_star = int(s.loc[i, "ts"])
            npv_star = float(s.loc[i, "NPV"])
            opt_rows.append({"case": case, "cn": cn, "ts_star": ts_star,
                             "NPV_max": round(npv_star, 2)})
            oil = df[(df["case"] == case) & (df["cn"] == cn) & (df["ts"] == ts_star)]
            oil = oil.sort_values("Production_Day")["Oil_Rate_m3_d"].to_numpy(float)
            p_be = eco.breakeven_oil_price(V_INJ, Q_INJ, F_CO2, cn, ts_star,
                                           TAU_INJ, oil, wor=WOR_DEFAULT)
            be_rows.append({"case": case, "cn": cn, "ts_star": ts_star,
                            "P_breakeven_10kCNY_per_m3": round(p_be, 4),
                            "P_breakeven_USD_per_bbl": round(p_be * 1e4 * 0.159 / 7.2, 1)})
    opt = pd.DataFrame(opt_rows)
    be = pd.DataFrame(be_rows)
    opt.to_csv(os.path.join(OUT_DIR, "T17_optimal_ts.csv"), index=False)

    # ---- WOR 敏感性（Case2, cn=1.0, ts*）----
    case, cn = "Case2", 1.0
    ts_star = int(opt[(opt["case"] == case) & (opt["cn"] == cn)]["ts_star"].iloc[0])
    oil = df[(df["case"] == case) & (df["cn"] == cn) & (df["ts"] == ts_star)]
    oil = oil.sort_values("Production_Day")["Oil_Rate_m3_d"].to_numpy(float)
    wor_sens = {w: round(eco.candidate_npv(V_INJ, Q_INJ, F_CO2, cn, ts_star,
                                           TAU_INJ, oil, wor=w)["NPV"], 2)
                for w in [0.0, 0.5, 1.0, 2.0]}

    # ---- 报告 ----
    L = []
    A = L.append
    A("T16/T17 NPV 基线、盈亏平衡与亏损口径  (2026-11-20)")
    A("=" * 64)
    A(f"[自检] 题面算例复算 = {total:.6f} 万元 (目标 848.25) → PASS")
    A("分项: " + ", ".join(f"{k}={v}" for k, v in parts.items()))
    A("")
    A(f"[口径] 基准操作: q_inj={Q_INJ:.0f} m3/d, V_inj={V_INJ:.0f} m3, "
      f"f_CO2={F_CO2}, tau_inj={TAU_INJ}, WOR={WOR_DEFAULT} (默认, 无产水数据)")
    A("[口径] 逐日折现 r=8%/yr, 折现基准第 0 天; 注入按实际天数计费; "
      "生产期不计日作业费")
    A("")
    A("[最优焖井时间] (经济口径: max NPV, 绝对值)")
    A(opt.to_string(index=False))
    A("")
    A("[盈亏平衡油价] (NPV=0, 绝对口径)")
    A(be.to_string(index=False))
    A("")
    A("[WOR 敏感性] Case2, cn=1.0, ts*=%d d: " % ts_star
      + ", ".join(f"WOR={w}: NPV={v}" for w, v in wor_sens.items())
      + "  (产水处理费占比小, 方向: WOR 越大 NPV 略降)")
    A("")
    A("[三类亏损口径] (T17, 分开报告不混用)")
    A("  ① 工程不可行: T10 的 10 条等效裂缝基准注入算例中，"
      "Case1 在 0.58 d 超压，Case2 在 6.60 d 超压。")
    neg = res[res["NPV"] < 0].groupby("case").size().to_dict()
    A(f"  ② 绝对 NPV<0 (本扫描): 各情景 NPV<0 的 (cn,ts) 组合数 = {neg}")
    A("  ③ 相对 B1 的 ΔNPV: 运行 13_b1_comparison.py，见 T17_b1_comparison.csv。"
      "B1 使用初始压力与饱和度，以及沿用 T22 储集长度的降阶生产反事实。")
    n15 = eco.materials(5000, 0.5, 1.5)
    A(f"[P3 提示] (V_inj=5000, f=0.5, C_n=1.5): m_nano={n15['m_nano']:.1f} t "
      f"> 30 t 上限 → 该组合优化中判不可行。")

    txt = "\n".join(L)
    with open(os.path.join(OUT_DIR, "T16_npv_selfcheck.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n\n--- T17 盈亏平衡 ---\n" + be.to_string(index=False))
    with open(os.path.join(OUT_DIR, "T17_breakeven_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)

    print(txt)
    return {"opt": opt.to_dict("records"), "be": be.to_dict("records")}


if __name__ == "__main__":
    main()
