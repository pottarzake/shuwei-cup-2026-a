# -*- coding: utf-8 -*-
"""
T35 全局敏感性分析（Sobol，Saltelli N=1024，含二阶指数）

前置交付：sobol_lib.py 纯 numpy 实现 Saltelli (2010) 采样与估计量（替代 SALib），
Ishigami 解析解自检通过后进入正题。

正演接口：model/p3_forward（T22 数据标定 Q1(ts)/Ld(ts) + V/f 缩放，
指数由 T31 标定为 ≈0，tau=0 基准）。model/ 目录只读，故本脚本以全向量化方式
复刻 evaluate() 的数学（_batch_eval），并在运行前用 600 组随机点对拍
p3_forward.evaluate 验证数值一致（max rel err < 1e-9），不改动 model/。

总纲 6.8 / Q21 口径：
  - Q360 分析：11 个输入（含 t_s），输出 360 d 累计油 Q_o；
  - t_s* 分析：外层 10 个输入（排除 t_s），每组样本在工程可行域内扫描
    t_s ∈ {5,...,60} d 取 NPV 最大者（并列取较短），记录无可行解比例；
    无可行解的外层样本以最短档 t_s=5 d 代入（如实报告比例）。

局限（如实报告）：
  - E、σ_H、ρ_nf 在当前降阶正演中为惰性参数（指数≈0）；
  - 待 T34 机理正演外壳接入后复跑（forward 接口不变，仅换模型）。

输出（03_结果与图表/）：
  T35_sobol_Q360_case{1,2,3}.csv / T35_sobol_tsstar_case{1,2,3}.csv
  T35_sobol_npvstar_case{1,2,3}.csv / T35_sobol_report.txt / T35_sobol.png
"""

import inspect
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "model"))

import sobol_lib
import p3_forward
import economics as eco

OUT_DIR = os.path.join(BASE, "..", "03_结果与图表")
os.makedirs(OUT_DIR, exist_ok=True)

N_SALTELLI = 1024
SEED = 2026
TS_GRID = np.arange(5.0, 61.0, 5.0)   # 焖井扫描档 12 档
_CHUNK = 4096                          # 批处理行数（内存控制）

# V/f 缩放指数：从 p3_forward.evaluate 的默认形参自动读取（T31 标定为 ≈0，
# 之后若再标定只需改 p3_forward，本脚本自动跟随；对拍自检兜底）。
_SIG = inspect.signature(p3_forward.evaluate)
_EXP_V = float(_SIG.parameters["exp_v"].default)
_EXP_F = float(_SIG.parameters["exp_f"].default)


# ---------------------------------------------------------------- 参数定义
# (名称, 下界, 上界, 形参含义)
PARAMS = [
    ("k_m0",    0.5, 2.0, "k_r"),     # 基质渗透率乘子
    ("phi_0",   0.8, 1.2, "phi_r"),   # 孔隙度乘子
    ("sigma_h", 0.7, 1.3, "sh_r"),    # 最小水平主应力乘子
    ("sigma_H", 0.8, 1.2, "inert"),   # 最大水平主应力（降阶模型未显式）
    ("E",       0.8, 1.2, "inert"),   # 杨氏模量（降阶模型未显式）
    ("C_n",     0.5, 1.5, "cn"),      # wt%
    ("V_inj",   1000.0, 5000.0, "V"),
    ("q_inj",   100.0, 800.0, "q"),
    ("f_CO2",   0.2, 0.8, "f"),
    ("rho_nf",  0.5, 1.5, "inert"),   # 纳米流体密度（降阶模型未显式）
]
PARAMS_TS = PARAMS + [("t_s", 5.0, 60.0, "ts")]
_COEF_COLS = ("Ld_a", "Ld_b", "Ld_c", "q_a", "q_b", "q_c")
_COEF = {}                            # case -> {col: (cn_sorted, val_sorted)}


def _init_coef():
    for case in (1, 2, 3):
        sub = p3_forward._TAB[p3_forward._TAB.case == case].sort_values("Cn")
        _COEF[case] = {c: (sub.Cn.values.astype(float), sub[c].values.astype(float))
                       for c in _COEF_COLS}


def _DF(d):
    """逐日折现因子 (1+r)^(-t/365)，与 economics._discount 同式。"""
    return (1.0 + eco.R_DISC) ** (-np.asarray(d, float) / 365.0)


def _batch_eval(case, cn, q, V, f, ts, k_r, phi_r, sh_r):
    """全向量化复刻 p3_forward.evaluate(tau=0, wor=0.5, 指数取其默认形参)。

    所有输入 (M,) 数组；返回 dict(Q_o, NPV, feasible, violation)。分块控内存。
    """
    cn = np.asarray(cn, float); q = np.asarray(q, float); V = np.asarray(V, float)
    f = np.asarray(f, float); ts = np.asarray(ts, float)
    k_r = np.asarray(k_r, float); phi_r = np.asarray(phi_r, float)
    sh_r = np.asarray(sh_r, float)
    M = cn.size
    out = {"Q_o": np.empty(M), "NPV": np.empty(M),
           "feasible": np.empty(M, bool), "violation": np.empty(M)}
    area, phi0, movable, qmax0 = p3_forward.CASE[case]
    for s0 in range(0, M, _CHUNK):
        e = min(s0 + _CHUNK, M)
        sl = slice(s0, e)
        c, qq, vv, ff, tt = cn[sl], q[sl], V[sl], f[sl], ts[sl]
        kr, pr, sr = k_r[sl], phi_r[sl], sh_r[sl]

        qmax = qmax0 * kr / np.sqrt(sr)
        phi = phi0 * pr
        t_inj = vv / qq

        # T22 标定系数（cn 线性插值，clip 到标定域，同 _row）
        cc = np.clip(c, _COEF[case]["q_a"][0][0], _COEF[case]["q_a"][0][-1])
        a_ = {col: np.interp(cc, *_COEF[case][col]) for col in _COEF_COLS}

        # 可行性（与 evaluate 同三项惩罚）
        m_nano = (c / 100.0) * eco.RHO_AQ * ((1.0 - ff) * vv) / 1000.0
        viol = np.maximum(qq - qmax, 0.0) / qmax \
            + np.maximum(t_inj + tt - 90.0, 0.0) / 90.0
        viol = viol + np.where(m_nano > eco.NANO_LIMIT_T + 1e-12,
                               np.maximum(m_nano - eco.NANO_LIMIT_T, 0.0)
                               / eco.NANO_LIMIT_T, 0.0)

        # 生产剖面（V/f 缩放指数由 p3_forward 默认形参决定，T31 后 ≈0）
        sv = (vv / 3000.0) ** _EXP_V
        sf = (ff / 0.5) ** _EXP_F
        q1 = (a_["q_a"] + a_["q_b"] * (1.0 - np.exp(-tt / a_["q_c"]))) \
            * sv * sf * np.sqrt(kr)
        Ld = (a_["Ld_a"] + a_["Ld_b"] * (1.0 - np.exp(-tt / a_["Ld_c"]))) * sv * sf
        Vo = area * Ld * phi * movable
        qw1 = 0.5 * q1
        scale = np.where(q1 + qw1 > 100.0,
                         100.0 / np.maximum(q1 + qw1, 1e-300), 1.0)
        q1s = q1 * scale
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = np.where(q1s > 0.0, Vo / np.maximum(q1s, 1e-300), np.inf)
            # 2026-11-20 T22 改版：油率为日内均值（闭合因子），日起点 0..359
            day_factor = np.where(np.isfinite(tau),
                                  -tau * np.expm1(-1.0 / tau), 1.0)
        t = np.arange(360.0)
        oil = q1s[:, None] * day_factor[:, None] \
            * np.exp(-t[None, :] / tau[:, None])
        oil = np.nan_to_num(oil)
        water = 0.5 * oil
        Q_o = oil.sum(axis=1)

        # NPV（candidate_npv 复刻，tau=0）
        m_co2 = eco.RHO_CO2_REF * (ff * vv) / 1000.0
        C_nano_v = eco.C_NANO * m_nano
        C_co2_v = eco.C_CO2 * m_co2
        a_inj = -(eco.C_INJ + (C_nano_v + C_co2_v) / t_inj)
        n_full = np.floor(t_inj + 1e-12).astype(int)
        frac = t_inj - n_full
        Kmax = int(n_full.max()) + 1
        cum = np.concatenate([[0.0], np.cumsum(_DF(np.arange(1, Kmax + 1.0)))])
        inj = a_inj * (cum[n_full]
                       + np.where(frac > 1e-9, frac * _DF(n_full + frac), 0.0))
        n_s = np.floor(tt + 1e-12).astype(int)
        fr_s = tt - n_s
        Ks = int(n_s.max()) + 1
        cum2 = np.concatenate([[0.0], np.cumsum(_DF(np.arange(1, Ks + 1.0)))])
        soak = -eco.C_SOAK * _DF(t_inj) \
            * (cum2[n_s] + np.where(fr_s > 1e-9, fr_s * _DF(n_s + fr_s), 0.0))
        start = t_inj + tt
        days = start[:, None] + np.arange(1.0, 361.0)[None, :]
        cf = (eco.P_OIL_STAR - eco.C_PROD) * oil - eco.C_WATER * water
        npv = -eco.C_FRAC + inj + soak + (cf * _DF(days)).sum(axis=1)

        out["Q_o"][sl] = Q_o
        out["NPV"][sl] = npv
        out["feasible"][sl] = viol <= 1e-12
        out["violation"][sl] = viol
    return out


def _selfcheck_replica(n=600):
    """对拍 _batch_eval 与 p3_forward.evaluate，返回 (ok, max_rel_err)。"""
    rng = np.random.default_rng(7)
    lo = np.array([p[1] for p in PARAMS_TS]); hi = np.array([p[2] for p in PARAMS_TS])
    X = lo + rng.random((n, len(PARAMS_TS))) * (hi - lo)
    worst = 0.0
    for case in (1, 2, 3):
        r = _batch_eval(case, X[:, 5], X[:, 7], X[:, 6], X[:, 8], X[:, 10],
                        X[:, 0], X[:, 1], X[:, 2])
        for j in range(n):
            ref = p3_forward.evaluate(
                case, X[j, 5], X[j, 7], X[j, 6], X[j, 8], X[j, 10], 0.0,
                k_r=X[j, 0], phi_r=X[j, 1], sh_r=X[j, 2])
            for k_out, k_ref in (("Q_o", "Q_o"), ("NPV", "NPV")):
                a, b = r[k_out][j], ref[k_ref]
                denom = max(abs(b), 1.0)
                worst = max(worst, abs(a - b) / denom)
            if r["feasible"][j] != bool(ref["feasible"]):
                worst = max(worst, 1.0)
    return worst < 1e-9, worst


# ---------------------------------------------------------------- 分析
def make_q360_forward(case):
    """Q360 分析 forward：X (M,11) -> Q_o (M,)。第 11 维是 t_s。"""
    def f(X):
        return _batch_eval(case, X[:, 5], X[:, 7], X[:, 6], X[:, 8], X[:, 10],
                           X[:, 0], X[:, 1], X[:, 2])["Q_o"]
    return f


def make_tsstar_forward(case):
    """t_s* 分析 forward：X (M,10) -> (ts*, NPV*, feasible)。"""
    G = TS_GRID.size

    def f(X):
        M = X.shape[0]
        cn = np.repeat(X[:, 5], G); q = np.repeat(X[:, 7], G)
        V = np.repeat(X[:, 6], G); ff = np.repeat(X[:, 8], G)
        kr = np.repeat(X[:, 0], G); pr = np.repeat(X[:, 1], G)
        sr = np.repeat(X[:, 2], G)
        ts = np.tile(TS_GRID, M)
        r = _batch_eval(case, cn, q, V, ff, ts, kr, pr, sr)
        npv = r["NPV"].reshape(M, G)
        feas = r["feasible"].reshape(M, G)
        npv_f = np.where(feas, npv, -np.inf)
        any_feas = feas.any(axis=1)
        best = np.argmax(npv_f, axis=1)      # 并列取首个（较短 ts）
        ts_star = TS_GRID[best]
        npv_star = npv[np.arange(M), best]
        ts_star = np.where(any_feas, ts_star, TS_GRID[0])
        npv_star = np.where(any_feas, npv_star, np.nan)
        return ts_star, npv_star, any_feas
    return f


def _save_csv(path, names, res, extra=None):
    d = {"param": names,
         "S1": np.round(res["S1"], 4),
         "S1_lo": np.round(res["ci"]["S1"][0], 4) if res["ci"] else np.nan,
         "S1_hi": np.round(res["ci"]["S1"][1], 4) if res["ci"] else np.nan,
         "ST": np.round(res["ST"], 4),
         "ST_lo": np.round(res["ci"]["ST"][0], 4) if res["ci"] else np.nan,
         "ST_hi": np.round(res["ci"]["ST"][1], 4) if res["ci"] else np.nan}
    if extra:
        d.update(extra)
    pd.DataFrame(d).to_csv(path, index=False)


def _top3(names, ST):
    idx = np.argsort(ST)[::-1][:3]
    return [(names[i], float(ST[i])) for i in idx]


def main():
    _init_coef()
    lines = ["T35 全局敏感性分析 Sobol (Saltelli N=%d, seed=%d, 含二阶)  (%s)"
             % (N_SALTELLI, SEED, time.strftime("%Y-%m-%d")),
             "=" * 68]
    ok, txt = sobol_lib.self_test()
    lines.append(txt)
    ok2, err = _selfcheck_replica()
    lines.append("[复刻正演对拍] 600 随机点 × 3 情景 vs p3_forward.evaluate: "
                 "max rel err = %.2e -> %s" % (err, "PASS" if ok2 else "FAIL"))
    if not (ok and ok2):
        lines.append("自检未通过，终止。")
        _write(lines)
        print("\n".join(lines))
        raise SystemExit(1)
    lines.append("")

    rng = np.random.default_rng(SEED)
    q360_bounds = np.array([[p[1], p[2]] for p in PARAMS_TS])
    tsstar_bounds = np.array([[p[1], p[2]] for p in PARAMS])
    names_q = [p[0] for p in PARAMS_TS]
    names_t = [p[0] for p in PARAMS]

    results = {}
    for case in (1, 2, 3):
        t0 = time.time()
        # ---- Q360 ----
        S = sobol_lib.saltelli_sample(q360_bounds, N_SALTELLI, rng=rng)
        yA, yB, yAB, yABr, yAB2 = sobol_lib.evaluate_matrices(
            make_q360_forward(case), S)
        r_q = sobol_lib.sobol_indices(yA, yB, yAB, yABr, yAB2, S["pairs"],
                                      n_boot=100, rng=rng)
        _save_csv(os.path.join(OUT_DIR, f"T35_sobol_Q360_case{case}.csv"),
                  names_q, r_q)
        results[(case, "Q360")] = r_q

        # ---- ts* / NPV*（同一批外层矩阵各评估一次，两个输出共用）----
        S2 = sobol_lib.saltelli_sample(tsstar_bounds, N_SALTELLI, rng=rng)
        f_ts = make_tsstar_forward(case)
        yA2, npvA, feasA = f_ts(S2["A"])
        yB2, npvB, _ = f_ts(S2["B"])
        AB_t = np.empty((S2["D"], N_SALTELLI)); AB_n = np.empty((S2["D"], N_SALTELLI))
        ABr_t = np.empty((S2["D"], N_SALTELLI)); ABr_n = np.empty((S2["D"], N_SALTELLI))
        for i in range(S2["D"]):
            AB_t[i], AB_n[i], _ = f_ts(S2["AB"][i])
            ABr_t[i], ABr_n[i], _ = f_ts(S2["ABr"][i])
        AB2_t = np.stack([f_ts(S2["AB2"][p])[0] for p in range(len(S2["pairs"]))])
        r_t = sobol_lib.sobol_indices(yA2, yB2, AB_t, ABr_t,
                                      AB2_t, S2["pairs"],
                                      n_boot=100, rng=rng)
        _save_csv(os.path.join(OUT_DIR, f"T35_sobol_tsstar_case{case}.csv"),
                  names_t, r_t)
        results[(case, "ts*")] = r_t

        # NPV* 的 Sobol（同一批外层样本，剔除无可行解行）
        ok_n = ~(np.isnan(npvA) | np.isnan(npvB)
                 | np.isnan(AB_n).any(axis=0))
        r_n = sobol_lib.sobol_indices(npvA[ok_n], npvB[ok_n],
                                      AB_n[:, ok_n], ABr_n[:, ok_n],
                                      n_boot=100, rng=rng)
        _save_csv(os.path.join(OUT_DIR, f"T35_sobol_npvstar_case{case}.csv"),
                  names_t, r_n)

        infeas_pct = 100.0 * float(np.mean(~feasA))
        lines.append(f"[Case{case}] Q360 一阶/总效应指数 (按 ST 排序前 6):")
        order = np.argsort(r_q["ST"])[::-1]
        for i in order[:6]:
            lines.append("  %-8s S1=%.3f ST=%.3f" % (names_q[i], r_q["S1"][i], r_q["ST"][i]))
        s2 = r_q["S2"]
        pairs = [(names_q[i], names_q[j], s2[i, j])
                 for i in range(s2.shape[0]) for j in range(i + 1, s2.shape[1])]
        pairs.sort(key=lambda x: -x[2])
        lines.append("  二阶 top3: " + "; ".join(
            f"{a}×{b}={v:.3f}" for a, b, v in pairs[:3]))
        lines.append(f"[Case{case}] t_s* 一阶/总效应指数 (前 6); "
                     f"外层无可行解比例 {infeas_pct:.1f}%:")
        if not np.isfinite(r_t["V"]) or r_t["V"] <= 0:
            lines.append("  外层样本输出方差为 0（可行域为空，t_s* 恒为最短档），"
                         "Sobol 分析不适用 —— 与 T29 'Case1 无可行解' 一致。")
        else:
            order = np.argsort(r_t["ST"])[::-1]
            for i in order[:6]:
                lines.append("  %-8s S1=%.3f ST=%.3f" % (names_t[i], r_t["S1"][i], r_t["ST"][i]))
            lines.append("  t_s* 主控 top3(ST): " + "; ".join(
                f"{n}={v:.3f}" for n, v in _top3(names_t, r_t["ST"])))
            pairs_t = [(names_t[i], names_t[j], r_t["S2"][i, j])
                       for i in range(r_t["S2"].shape[0])
                       for j in range(i + 1, r_t["S2"].shape[1])]
            pairs_t.sort(key=lambda x: -x[2])
            lines.append("  t_s* 二阶 top3: " + "; ".join(
                f"{a}×{b}={v:.3f}" for a, b, v in pairs_t[:3]))
        lines.append(f"  用时 {time.time() - t0:.0f} s")
        lines.append("")

    lines.append("与 P4-04 对照（最优焖井时间主控排序: 注入液量 > 毛管压倍数 > "
                 "基质渗透率 > 孔隙度 > ...）:")
    for case in (1, 2, 3):
        r_t = results[(case, "ts*")]
        if not np.isfinite(r_t["V"]) or r_t["V"] <= 0:
            lines.append(f"  Case{case}: 可行域为空，无 t_s* 排序（见上）")
        else:
            lines.append(f"  Case{case}: " + "; ".join(
                f"{n}={v:.3f}" for n, v in _top3(names_t, r_t["ST"])))
    lines.append("局限: E、sigma_H、rho_nf 在降阶正演中为惰性参数(指数≈0);"
                 " tau=0 基准; V/f 缩放指数已按 T31 标定取 %.3f/%.3f;"
                 " 待 T34 机理外壳接入后复跑。" % (_EXP_V, _EXP_F))
    _write(lines)
    _plot(results, names_q, names_t)
    print("\n".join(lines))


def _write(lines):
    with open(os.path.join(OUT_DIR, "T35_sobol_report.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def _plot(results, names_q, names_t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))
    for col, case in enumerate((1, 2, 3)):
        for row, (key, names) in enumerate((("Q360", names_q), ("ts*", names_t))):
            r = results[(case, key)]
            ax = axes[row, col]
            x = np.arange(len(names))
            ax.bar(x - 0.2, r["S1"], 0.4, label="$S_i$", color="#1f77b4")
            ax.bar(x + 0.2, r["ST"], 0.4, label="$ST_i$", color="#d62728")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"Case{case} — {'$Q_{360}$' if key == 'Q360' else '$t_s^*$'}",
                         fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
            if row == 1 or col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("T35 Sobol 一阶/总效应指数 (Saltelli N=1024, 降阶正演 p3_forward)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "T35_sobol.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
