# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
"""
data_model.py —— P4 数据模块（队友负责，2026-11-20）

职责：
  1. 读取附件 C，按"情景 × 浓度 × 焖井时间"整组划分训练/验证/测试（禁止组内拆分）；
  2. 提供组级汇总表（Q360、初始产量等）供 P4 与 NPV 使用；
  3. 基线模型：常数预测器 + 线性回归（numpy 实现，无 sklearn 依赖）。

划分口径（v0.3 任务卡 T03，写死勿改）：
  训练 train ：t_s ∈ {5,10,20,25,30,35,40,50}   8 档 × 3 情景 × 3 浓度 = 72 组
  验证 val   ：t_s = 55                          9 组
  测试 test  ：t_s ∈ {15,45,60}                   27 组
                 其中 15/45 ∈ 训练区间内  → 区间内留档（interpolation）
                      60  > 所有训练档      → 外推（extrapolation）
  三组 (case, cn, ts) 无重叠；测试含未参与训练的焖井档（15/45/60 全部）。

运行：python data_model.py  （在 02_代码/ 下运行；依赖 numpy、pandas）
输出：../03_结果与图表/T03_split_groups.csv
      ../03_结果与图表/T03_baseline_report.txt
      ../03_结果与图表/T03_split_overview.png
"""

import os
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_XLSX = os.path.join(WS, "01_数据", "附件C_Synthetic_Production_Data.xlsx")
OUT_DIR = os.path.join(WS, "03_结果与图表")

CASES = ["Case1_low-permeability high-stress",
         "Case2_medium-permeability low-stress",
         "Case3_high-permeability complex fractures"]
CASE_SHORT = {c: f"Case{i+1}" for i, c in enumerate(CASES)}

TS_TRAIN = [5, 10, 20, 25, 30, 35, 40, 50]
TS_VAL = [55]
TS_TEST = [15, 45, 60]
TS_TEST_INTERP = [15, 45]   # 区间内留档
TS_TEST_EXTRAP = [60]       # 外推


def load_data():
    """读取附件 C 原始日度数据。"""
    df = pd.read_excel(DATA_XLSX)
    df["case"] = df["Case_ID"].map(CASE_SHORT)
    return df


def group_table(df):
    """组级汇总：每组一行，Q360 取第 360 天 Cum_Oil（含噪声，题面允许直接使用）。

    注意（附件 C 使用规则）：不得由 Cum_Oil 反推日产油再加噪；这里只取 Q360。
    """
    g = df[df["Production_Day"] == 360].copy()
    g = g.rename(columns={"Nano_Concentration_wt%": "cn", "Soak_Time_Days": "ts"})
    g = g[["case", "cn", "ts", "Cum_Oil_m3"]].rename(columns={"Cum_Oil_m3": "Q360"})
    g = g.sort_values(["case", "cn", "ts"]).reset_index(drop=True)

    q1 = (df[df["Production_Day"] == 1]
          .groupby(["case", "Nano_Concentration_wt%", "Soak_Time_Days"])["Oil_Rate_m3_d"]
          .first()
          .rename_axis(["case", "cn", "ts"]))
    g = g.merge(q1.rename("q_day1"), on=["case", "cn", "ts"], how="left")
    return g


def add_split(g):
    """按 ts 打 split 标签：train / val / test_interp / test_extrap。"""
    def lab(ts):
        if ts in TS_TRAIN:
            return "train"
        if ts in TS_VAL:
            return "val"
        if ts in TS_TEST_INTERP:
            return "test_interp"
        return "test_extrap"
    g = g.copy()
    g["split"] = g["ts"].map(lab)
    return g


def check_split(g):
    """硬性检查：三组无 (case,cn,ts) 重叠；训练/验证/测试组数正确。"""
    n = g["split"].value_counts().to_dict()
    keys = g.groupby(["case", "cn", "ts"]).size()
    ok_overlap = bool((keys <= 1).all())
    ok_counts = (n.get("train") == 72 and n.get("val") == 9
                 and n.get("test_interp") == 18 and n.get("test_extrap") == 9)
    return {"counts": n, "no_overlap": ok_overlap, "counts_ok": ok_counts}


# ---------------- 基线模型（numpy，无 sklearn） ----------------

def build_features(g, style="zg"):
    """P4 模型统一特征入口（2026-11-20 起全部脚本必须走这里，勿各自实现）。

    style="zg"（默认，总纲 6.3 口径）：[C_n, t_s, C_n·t_s, log t_s, I(Case2), I(Case3)]
    style="simple"（早期对照）：[C_n, t_s, I(Case2), I(Case3)]
    返回 (X, y)。
    """
    cn = g["cn"].to_numpy(float)
    ts = g["ts"].to_numpy(float)
    i2 = (g["case"] == "Case2").astype(float)
    i3 = (g["case"] == "Case3").astype(float)
    if style == "zg":
        X = np.column_stack([cn, ts, cn * ts, np.log(ts), i2, i3])
    elif style == "simple":
        X = np.column_stack([cn, ts, i2, i3])
    else:
        raise ValueError(style)
    return X, g["Q360"].to_numpy(float)


def _design(g):
    """特征：[1, cn, ts, I(Case2), I(Case3)]；另附 log(ts) 增强版供 T12 参考。"""
    X = np.column_stack([
        np.ones(len(g)),
        g["cn"].to_numpy(float),
        g["ts"].to_numpy(float),
        (g["case"] == "Case2").astype(float),
        (g["case"] == "Case3").astype(float),
    ])
    Xlog = np.column_stack([X, np.log(g["ts"].to_numpy(float))])
    return X, Xlog


def fit_baseline(g_train):
    """常数预测器（分情景均值）+ 线性回归。返回 dict of predictors。"""
    case_mean = g_train.groupby("case")["Q360"].mean().to_dict()
    global_mean = float(g_train["Q360"].mean())

    X, Xlog = _design(g_train)
    y = g_train["Q360"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef_log, *_ = np.linalg.lstsq(Xlog, y, rcond=None)

    def pred_const(g):
        return g["case"].map(case_mean).fillna(global_mean).to_numpy(float)

    def pred_lin(g):
        Xg, _ = _design(g)
        return Xg @ coef

    def pred_log(g):
        _, Xg = _design(g)
        return Xg @ coef_log

    return {"const": pred_const, "lin": pred_lin, "loglin": pred_log,
            "case_mean": case_mean, "global_mean": global_mean,
            "coef": coef.tolist(), "coef_log": coef_log.tolist()}


def rmse_mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return rmse, mape


def evaluate_baselines(g):
    """在 val / test_interp / test_extrap 上评估三个基线。"""
    models = fit_baseline(g[g["split"] == "train"])
    rows = []
    for split in ["val", "test_interp", "test_extrap"]:
        gs = g[g["split"] == split]
        for name, fn in models.items():
            if name in ("case_mean", "global_mean", "coef", "coef_log"):
                continue
            y = gs["Q360"].to_numpy(float)
            yp = fn(gs)
            r, m = rmse_mape(y, yp)
            rows.append({"split": split, "model": name, "n": len(gs),
                         "rmse": round(r, 2), "mape_pct": round(m, 2)})
    return pd.DataFrame(rows), models


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    g = add_split(group_table(df))
    g.to_csv(os.path.join(OUT_DIR, "T03_split_groups.csv"), index=False)

    checks = check_split(g)
    report, models = evaluate_baselines(g)

    lines = []
    lines.append("T03 数据划分与基线  (2026-11-20)")
    lines.append("=" * 60)
    lines.append(f"原始数据: {df.shape[0]} 行 × {df.shape[1]} 列; "
                 f"null={int(df.isna().sum().sum())}, 完全重复行={int(df.duplicated().sum())}")
    lines.append(f"总组数: {len(g)}  (3 case × 3 cn × 12 ts)")
    lines.append(f"划分组数: {checks['counts']}")
    lines.append(f"无 (case,cn,ts) 重叠: {checks['no_overlap']}")
    lines.append(f"组数符合 72/9/18+9: {checks['counts_ok']}")
    lines.append("")
    lines.append("常数预测器（分情景训练均值）:")
    for c, v in models["case_mean"].items():
        lines.append(f"  {c}: {v:.1f} m3")
    lines.append(f"  全局均值: {models['global_mean']:.1f} m3")
    lines.append("")
    lines.append("基线评估（目标 Q360, 单位 m3）:")
    lines.append(report.to_string(index=False))
    lines.append("")
    lines.append("说明: test_interp = t_s∈{15,45} 区间内留档; "
                 "test_extrap = t_s=60 外推; loglin 为含 log(ts) 的线性参考变体。")

    txt = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "T03_baseline_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

    # 图：Q360 ~ ts，按 split 着色（英文标注，后续可直接进论文）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    style = {"train": ("o", "#1f77b4", "train"),
             "val": ("s", "#2ca02c", "val"),
             "test_interp": ("^", "#ff7f0e", "test (interp.)"),
             "test_extrap": ("D", "#d62728", "test (extrap.)")}
    for ax, cn in zip(axes, sorted(g["cn"].unique())):
        for case in ["Case1", "Case2", "Case3"]:
            sub = g[(g["cn"] == cn) & (g["case"] == case)]
            for split, (mk, col, lab) in style.items():
                s = sub[sub["split"] == split]
                ax.plot(s["ts"], s["Q360"], mk, color=col, ms=5,
                        label=lab if case == "Case1" else None)
        ax.set_title(f"C_n = {cn} wt%")
        ax.set_xlabel("Soak time (d)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cumulative oil at 360 d (m$^3$)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "T03_split_overview.png"), dpi=300)
    plt.close(fig)

    print(txt)
    return {"checks": checks, "report": report.to_dict("records")}


if __name__ == "__main__":
    main()
