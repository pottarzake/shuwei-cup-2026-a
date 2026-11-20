# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
"""
08_model_comparison.py —— T12 P4 多模型对比（2026-11-20）

四模型：二次响应面 / GBM(限深3) / 随机森林 / 高斯过程（ml_lib.py，numpy 手写，
因本机无法安装 scikit-learn）。

协议：
  - 目标：组级 Q360；特征 X = [C_n, t_s, I(Case2), I(Case3)]
  - 训练集 = 8 档焖井训练组（72 组）；GroupKFold 按 t_s 整组折叠（8 折，
    留一整档焖井在外，镜像题目的外推检验，防泄漏；方法论引 P4-01）
  - 验证集（t_s=55）用于对比与后续 T13 选模型；测试集（15/45/60）T12 不碰
输出：T12_model_comparison.txt / T12_model_comparison.png / T12_cv_predictions.csv
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


def make_models():
    return {
        "quadratic": ml_lib.QuadraticResponseSurface(),
        "GBM": ml_lib.GradientBoosting(n_estimators=300, learning_rate=0.05,
                                       max_depth=3, min_samples_leaf=3),
        "RF": ml_lib.RandomForest(n_estimators=300, max_depth=5,
                                  min_samples_leaf=2),
        "GP": ml_lib.GaussianProcessRBF(),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = data_model.load_data()
    g = data_model.add_split(data_model.group_table(df))
    gtr = g[g["split"] == "train"].reset_index(drop=True)
    gval = g[g["split"] == "val"].reset_index(drop=True)
    Xtr, ytr = features(gtr)
    Xval, yval = features(gval)

    rows, fitted, cv_pred_rows = [], {}, []
    for name, model in make_models().items():
        # GroupKFold by ts
        rms, mps, sp = [], [], []
        for tr, te in ml_lib.group_kfold(gtr["ts"].to_numpy(), 8):
            m = make_models()[name]
            m.fit(Xtr[tr], ytr[tr])
            p = m.predict(Xtr[te])
            rms.append(ml_lib.rmse(ytr[te], p))
            mps.append(ml_lib.mape(ytr[te], p))
            sp.append(ml_lib.spearman(ytr[te], p))
            for i, pi in zip(te, p):
                cv_pred_rows.append({"model": name, "case": gtr.loc[i, "case"],
                                     "cn": gtr.loc[i, "cn"],
                                     "ts": gtr.loc[i, "ts"],
                                     "actual": ytr[i], "pred": pi})
        fitted[name] = model.fit(Xtr, ytr)
        pv = model.predict(Xval)
        rows.append({"model": name,
                     "cv_rmse_mean": round(np.mean(rms), 1),
                     "cv_rmse_max": round(np.max(rms), 1),
                     "cv_mape_mean": round(np.mean(mps), 2),
                     "cv_spearman": round(np.nanmean(sp), 3),
                     "val_rmse": round(ml_lib.rmse(yval, pv), 1),
                     "val_mape": round(ml_lib.mape(yval, pv), 2),
                     "val_spearman": round(ml_lib.spearman(yval, pv), 3)})

    rep = pd.DataFrame(rows).sort_values("cv_rmse_mean")
    rep.to_csv(os.path.join(OUT_DIR, "T12_model_comparison.csv"), index=False)
    pd.DataFrame(cv_pred_rows).to_csv(
        os.path.join(OUT_DIR, "T12_cv_predictions.csv"), index=False)

    # ---- 图：左 CV/val RMSE 条形，右 最优模型 val 预测-实际 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    ax = axes[0]
    xpos = np.arange(len(rep))
    ax.bar(xpos - 0.2, rep["cv_rmse_mean"], 0.38, label="CV RMSE (mean)")
    ax.bar(xpos + 0.2, rep["val_rmse"], 0.38, label="Val RMSE ($t_s$=55)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(rep["model"])
    ax.set_ylabel("RMSE (m$^3$)")
    ax.set_title("(a) Model comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    best_name = rep.iloc[0]["model"]
    best = fitted[best_name]
    pv = best.predict(Xval)
    ax = axes[1]
    for case, col in [("Case1", "#d62728"), ("Case2", "#1f77b4"),
                      ("Case3", "#2ca02c")]:
        m = gval["case"] == case
        ax.plot(yval[m], pv[m], "o", ms=6, color=col, label=case)
    lim = [0, max(yval.max(), pv.max()) * 1.1]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual $Q_{360}$ (m$^3$)")
    ax.set_ylabel("Predicted $Q_{360}$ (m$^3$)")
    ax.set_title(f"(b) Best model ({best_name}) on validation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "T12_model_comparison.png"), dpi=300)
    plt.close(fig)

    # ---- 文字报告 ----
    L = []
    A = L.append
    A("T12 P4 多模型对比  (2026-11-20)")
    A("=" * 60)
    A("协议: 训练 8 档焖井 72 组; GroupKFold 按 t_s 整组 8 折(留一整档在外);")
    A("验证 t_s=55; 测试集(15/45/60)本步不碰。目标 Q360, 特征 [cn, ts, case].")
    A("注: sklearn 因网络限速无法安装, 四模型为 numpy 手写实现(ml_lib.py),")
    A("   GBM 限深 3(任务卡防过拟合口径)。")
    A("")
    A(rep.to_string(index=False))
    A("")
    A(f"最优模型(按 CV RMSE): {best_name}")
    A("解读: GBM/RF 捕捉饱和非线性; 二次面欠拟合 t_s 饱和; GP 给预测区间(T14 用)。")
    txt = "\n".join(L)
    with open(os.path.join(OUT_DIR, "T12_model_comparison.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    return {"rep": rep.to_dict("records")}


if __name__ == "__main__":
    main()
