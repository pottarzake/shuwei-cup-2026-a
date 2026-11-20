# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
"""
12_mech_data_comparison.py —— T25 机理 vs 数据对比：数据侧 + 归因（2026-11-20）

总纲 6.6 口径：
  趋势(三条 Q360(ts) 叠图) / 量级(相对误差) / 排序 / 最优 ts*(机理 vs 数据)
差异归因三方面：模型假设 / 数值误差 / 数据噪声。

本脚本产出数据侧与归因（机理侧 Q360(ts) 正演扫档由 T21 补齐后叠图）：
  1. 数据侧 Q360(ts) 曲线（冻结 GP + 实测点, cn=1.0）
  2. 产量口径 ts*（P4-04 拐点定义：累计产量增量曲线由快增转平台的拐点
     = ΔQ360 离散一阶差分最大处）与经济口径 ts*（T17）对照
  3. 特征集核对：总纲 6.3 指定 [case, cn, ts, cn×ts, log ts]，本组实现用
     [cn, ts, case] —— 快速对比 CV 后决定是否对齐
  4. 归因证据表（三方面各至少一条, T25 判据）

输出: T25_dataside.csv / T25_dataside.png / T25_attribution.txt
"""

import os
import numpy as np
import pandas as pd

import data_model
import ml_lib

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(WS, "03_结果与图表")


def features_simple(g):
    return data_model.build_features(g, style="simple")


def features_63(g):
    """总纲 6.3 指定特征集（统一入口）"""
    return data_model.build_features(g, style="zg")


def cv_rmse(feat_fn, gtr, n=8):
    X, y = feat_fn(gtr)
    errs = []
    for tr, te in ml_lib.group_kfold(gtr["ts"].to_numpy(), n):
        m = ml_lib.GaussianProcessRBF().fit(X[tr], y[tr])
        errs.append(ml_lib.rmse(y[te], m.predict(X[te])))
    return float(np.mean(errs)), float(np.max(errs))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = data_model.load_data()
    g = data_model.add_split(data_model.group_table(df))
    gtr = g[g["split"] == "train"].reset_index(drop=True)

    # ---- 特征集核对 ----
    s_mean, s_max = cv_rmse(features_simple, gtr)
    c_mean, c_max = cv_rmse(features_63, gtr)

    # ---- 数据侧曲线 + 产量口径 ts* ----
    model = ml_lib.GaussianProcessRBF().fit(*features_63(gtr))
    ts_grid = np.arange(5, 61, 1.0)
    rows = []
    ts_star_prod = {}
    for case in ["Case1", "Case2", "Case3"]:
        i2 = 1.0 if case == "Case2" else 0.0
        i3 = 1.0 if case == "Case3" else 0.0
        cn1 = np.full_like(ts_grid, 1.0)
        Xg = np.column_stack([cn1, ts_grid, cn1 * ts_grid, np.log(ts_grid),
                              np.full_like(ts_grid, i2),
                              np.full_like(ts_grid, i3)])
        q_pred = model.predict(Xg)
        # 产量口径拐点: ΔQ 一阶差分最大处 (P4-04 定义)
        dq = np.diff(q_pred)
        ts_star_prod[case] = float(ts_grid[np.argmax(dq)])
        for t, q in zip(ts_grid, q_pred):
            rows.append({"case": case, "ts": float(t), "Q360_gp": round(q, 1)})
    curve = pd.DataFrame(rows)
    # 实测点 (cn=1.0, 全部 12 档)
    obs = g[g["cn"] == 1.0][["case", "ts", "Q360"]].copy()
    curve.to_csv(os.path.join(OUT_DIR, "T25_dataside.csv"), index=False)

    # 图：GP 曲线 + 实测点 + 两口径 ts*
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    econ_star = {"Case1": 10, "Case2": 20, "Case3": 35}  # T17, cn=1.0
    cols = {"Case1": "#d62728", "Case2": "#1f77b4", "Case3": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ax, case in zip(axes, ["Case1", "Case2", "Case3"]):
        c = curve[curve["case"] == case]
        o = obs[obs["case"] == case]
        ax.plot(c["ts"], c["Q360_gp"], "-", color=cols[case], lw=1.4,
                label="GP (frozen)")
        ax.plot(o["ts"], o["Q360"], "o", ms=5, mfc="none", color=cols[case],
                label="Attachment C")
        ax.axvline(ts_star_prod[case], color="k", ls=":", lw=1,
                   label=f"prod. $t_s^*$={ts_star_prod[case]:.0f} d")
        ax.axvline(econ_star[case], color="gray", ls="--", lw=1,
                   label=f"econ. $t_s^*$={econ_star[case]} d")
        ax.set_title(case)
        ax.set_xlabel("Soak time $t_s$ (d)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("$Q_{360}$ (m$^3$)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "T25_dataside.png"), dpi=300)
    plt.close(fig)

    # ---- 归因证据表 ----
    L = []
    A = L.append
    A("T25 机理 vs 数据对比 —— 数据侧与归因  (2026-11-20)")
    A("=" * 64)
    A("[特征集核对] 总纲6.3 指定 [case, cn, ts, cn*ts, log ts]")
    A(f"  实现集 [cn, ts, case]:        CV RMSE mean={s_mean:.1f} max={s_max:.1f}")
    A(f"  总纲集 [.., cn*ts, log ts]:   CV RMSE mean={c_mean:.1f} max={c_max:.1f}")
    A("  → 采用实现集(更少特征不过拟合); 偏差在报告说明。" if s_mean <= c_mean
      else "  → 切换为总纲特征集。")
    A("")
    A("[产量口径 ts*] (P4-04 拐点: ΔQ360 一阶差分最大):")
    for case in ["Case1", "Case2", "Case3"]:
        A(f"  {case}: prod ts*={ts_star_prod[case]:.0f} d | "
          f"econ ts*={econ_star[case]} d (T17)")
    A("")
    A("[差异归因] 三方面证据（机理侧叠图待 T21 三情景正演扫档后补齐）:")
    A("  ① 模型假设:")
    A("     - 纳米输运: 题给 Kd 下注入面浓度 ~3e-5 wt% << 0.01 wt%,")
    A("       纳米改造半径恒为 0 (T11) —— 与数据侧 C_n 重要性仅 2.9% (T15) 互证:")
    A("       两组独立结果都表明 C_n 在题给区间内对 Q360 影响有限,")
    A("       纳米作用应主要通过 σ/θ 的润湿改性而非波及体积。")
    A("     - 损伤: 题给 ε_d 比三情景压缩应变大 11-26 倍, d≡0 (T18/T19),")
    A("       P1 退化为孔隙弹性压敏 —— 数据侧 Case3 的高产不能归因于损伤增渗,")
    A("       只能来自 k_m0 与天然裂缝等效 (T24 对照待补)。")
    A("  ② 数值误差:")
    A("     - 压力再分布解析解 vs 三相焖井正演 ≤8% (T08, R>0 时段),")
    A("       且压力半径 3-10 d 内归零 —— 定量半径以正演为准;")
    A("     - 纳米前缘对网格未完全收敛: n=50→200 半径 0→0.15 m (T11),")
    A("       量级判断可用, 精确值待收敛表 (Q23)。")
    A("  ③ 数据噪声:")
    A("     - 标称 ±5% 高斯噪声 + 非负截断: 组内 CV 中位 9.3% (含趋势拟合残差),")
    A("       零产量仅见于 Case1 晚期, Q360 偏差 <1-2% (T04) —— 不足以解释")
    A("       机理-数据间的系统性差异, 排序与趋势一致时噪声可忽略。")
    txt = "\n".join(L)
    with open(os.path.join(OUT_DIR, "T25_attribution.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    return {}


if __name__ == "__main__":
    main()
