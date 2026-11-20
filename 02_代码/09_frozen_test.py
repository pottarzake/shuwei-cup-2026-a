# -*- coding: utf-8 -*-
"""
09_frozen_test.py —— T13 模型选择与冻结测试（2026-11-20）

选择（依据 T12）：GP（CV RMSE 最低 56.6、val MAPE 1.61%、自带预测区间）。
冻结：特征 [cn, ts, I(Case2), I(Case3)]，超参网格同 ml_lib.GaussianProcessRBF。
协议：只用训练 72 组拟合；测试集（15/45 区间留档 + 60 外推）只评估这一次。
输出：T13_frozen_test.txt / T13_test_predictions.csv / T13_frozen_test.png
"""

import os
import numpy as np
import pandas as pd

import data_model
import ml_lib

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(WS, "03_结果与图表")


def features(g):
    # 统一走总纲 6.3 特征集（2026-11-20 起）
    return data_model.build_features(g, style="zg")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = data_model.load_data()
    g = data_model.add_split(data_model.group_table(df))
    gtr = g[g["split"] == "train"]
    model = ml_lib.GaussianProcessRBF().fit(*features(gtr))

    rows = []
    for split, lab in [("test_interp", "interp (t_s=15/45)"),
                       ("test_extrap", "extrap (t_s=60)")]:
        gs = g[g["split"] == split]
        X, y = features(gs)
        p = model.predict(X)
        for i in range(len(gs)):
            rows.append({"split": split, "case": gs.iloc[i]["case"],
                         "cn": gs.iloc[i]["cn"], "ts": int(gs.iloc[i]["ts"]),
                         "actual": y[i], "pred": round(p[i], 1)})
        print(f"[{lab}] n={len(gs)} RMSE={ml_lib.rmse(y, p):.1f} "
              f"MAPE={ml_lib.mape(y, p):.2f}% "
              f"Spearman={ml_lib.spearman(y, p):.3f}")
        for case in ["Case1", "Case2", "Case3"]:
            m = gs["case"] == case
            if m.sum():
                print(f"    {case}: n={int(m.sum())} RMSE={ml_lib.rmse(y[m.values], p[m.values]):.1f} "
                      f"MAPE={ml_lib.mape(y[m.values], p[m.values]):.2f}%")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "T13_test_predictions.csv"), index=False)

    # 图：测试集预测-实际（按 split 标记）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    style = {"test_interp": ("o", "#1f77b4"), "test_extrap": ("D", "#d62728")}
    for split, (mk, col) in style.items():
        s = res[res["split"] == split]
        ax.plot(s["actual"], s["pred"], mk, ms=6, color=col,
                label=split.replace("test_", ""))
    lim = [0, res["actual"].max() * 1.1]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual $Q_{360}$ (m$^3$)")
    ax.set_ylabel("Predicted $Q_{360}$ (m$^3$)")
    ax.set_title("Frozen GP: one-shot test evaluation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "T13_frozen_test.png"), dpi=300)
    plt.close(fig)

    L = ["T13 模型选择与冻结测试  (2026-11-20)", "=" * 60,
         "选定模型: GP (T12 依据: CV RMSE 25.1 最低, val MAPE 1.59%)",
         "冻结: X=[cn, ts, cn*ts, log ts, I(Case2), I(Case3)] (总纲6.3); 超参网格同 T12; 只用训练 72 组拟合",
         "协议: 测试集只评估这一次 (15/45 区间留档, 60 外推)"]
    for split, lab in [("test_interp", "区间内留档 t_s=15/45"),
                       ("test_extrap", "外推 t_s=60")]:
        gs = res[res["split"] == split]
        L.append(f"[{lab}] n={len(gs)} RMSE={ml_lib.rmse(gs['actual'], gs['pred']):.1f} "
                 f"MAPE={ml_lib.mape(gs['actual'], gs['pred']):.2f}% "
                 f"Spearman={ml_lib.spearman(gs['actual'], gs['pred']):.3f}")
    L.append("注: 外推误差大属预期(题目本身要求检验外推), 如实报告。")
    txt = "\n".join(L)
    with open(os.path.join(OUT_DIR, "T13_frozen_test.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    return {}


if __name__ == "__main__":
    main()
