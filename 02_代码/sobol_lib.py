# -*- coding: utf-8 -*-
"""
Sobol 全局敏感性分析库（T35 前置，纯 numpy 手写，替代 SALib）。

本机 pip 装不上 SALib（网络限速），按 Saltelli 等 (2010) 的样本设计与
估计量手写实现。三套矩阵，方向相反，用途不同（这是最易错的点）：

  A_B^(i)    = B 的第 i 列换成 A 的第 i 列     → 一阶指数
  A_B^(~i)   = A 的第 i 列换成 B 的第 i 列     → 总效应指数 (Jansen)
  A_B^(ij)   = B 的第 i、j 列换成 A 的第 i、j 列 → 二阶指数

推导（f_S 为 ANOVA 分量，E[f_S f_T]=0 (S≠T)，对任一变元边缘化得 0）：
  E[f(A) f(A_B^i)]     = f0^2 + V_i            （一阶）
  E[f(A) f(A_B^{ij})]  = f0^2 + V_i + V_j + V_ij （二阶）
  E[(f(A) - f(A_B^{~i}))^2] = 2 (V - V_{-i})  （总效应，i 以外分量全抵消）

估计量（Saltelli 1993 一阶；Jansen 1999 总效应；Saltelli 扩展二阶）：
  V   = (1/N)Σ (f(A) - f̄)^2            （各矩阵先减自身样本均值，防大数相消）
  S_i   = (1/N)Σ (f(A)-f̄)(f(A_B^i)-f̄_i) / V
  ST_i  = (1/(2N))Σ ((f(A)-f̄) - (f(A_B^{~i})-f̄_{~i}))^2 / V
  S_ij  = (1/N)Σ (f(A_B^{ij})-f̄_{ij})(f(A)-f̄) / V - S_i - S_j

反例备忘（曾在此踩坑，勿再误用）：
  - mean(f(B)·(f(A_B^i)−f(A)))/V 不是一阶估计量；
  - mean((f(A)−f(A_B^i))^2)/(2V) 用错矩阵方向时给出 1−S_i 而非 ST_i；
  - A_B^i × A_B^j 乘积型不含 V_ij 项（E 只剩 k∉{i,j} 分量）。

总评估次数：N(2 + 2D + D(D-1)/2)。

自检：Ishigami f = sin x1 + A sin^2 x2 + B x3^4 sin x1 (A=7, B=0.1)
解析解: S1 = [0.3139, 0.4424, 0], ST = [0.5576, 0.4424, 0.2437],
        S12 = 0, S13 ≈ 0.2437, S23 = 0。
"""

import numpy as np


def saltelli_sample(bounds, N, rng=None, second_order=True):
    """生成 Saltelli 样本矩阵。bounds: (D,2) 各参数 [lo,hi]。

    返回 dict: A, B (N,D), AB (D,N,D)（i 列来自 A）,
    ABr (D,N,D)（i 列来自 B，总效应用）, AB2 (P,N,D) 与 pairs (P,2)（二阶）。
    总评估次数 N(2+2D+P)，P = D(D-1)/2。
    """
    rng = np.random.default_rng(rng)
    lo = bounds[:, 0]; hi = bounds[:, 1]
    D = bounds.shape[0]
    A = lo + rng.random((N, D)) * (hi - lo)
    B = lo + rng.random((N, D)) * (hi - lo)
    AB = np.empty((D, N, D))
    ABr = np.empty((D, N, D))
    for i in range(D):
        M = B.copy(); M[:, i] = A[:, i]
        AB[i] = M
        R = A.copy(); R[:, i] = B[:, i]
        ABr[i] = R
    out = {"A": A, "B": B, "AB": AB, "ABr": ABr, "N": int(N), "D": int(D),
           "AB2": None, "pairs": None}
    if second_order:
        pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]
        AB2 = np.empty((len(pairs), N, D))
        for p, (i, j) in enumerate(pairs):
            M = B.copy()
            M[:, i] = A[:, i]
            M[:, j] = A[:, j]
            AB2[p] = M
        out["AB2"] = AB2
        out["pairs"] = pairs
    return out


def evaluate_matrices(f, S):
    """对全部样本矩阵求模型输出。f: (M, D) -> (M,) 按行批量。

    返回 yA (N,), yB (N,), yAB (D,N), yABr (D,N), yAB2 (P,N) 或 None。
    """
    yA = np.asarray(f(S["A"]), float)
    yB = np.asarray(f(S["B"]), float)
    yAB = np.stack([np.asarray(f(S["AB"][i]), float) for i in range(S["D"])])
    yABr = np.stack([np.asarray(f(S["ABr"][i]), float) for i in range(S["D"])])
    if S["AB2"] is not None:
        yAB2 = np.stack([np.asarray(f(S["AB2"][p]), float)
                         for p in range(S["AB2"].shape[0])])
    else:
        yAB2 = None
    return yA, yB, yAB, yABr, yAB2


def _est(yA, yB, yAB, yABr, yAB2, pairs):
    """单次（重采样）Sobol 指数估计。

    各矩阵先减自身样本均值再算矩：期望不变，但避免 f0^2 >> V 时的
    大数相消（本问题 Q360 ~ 3e3、f0^2 ~ 9e6 >> V_i ~ 1e5，
    不减均值时一阶指数会被 MC 噪声淹没，实测 S1 和 >2 且 k_m0 纯噪声 0.227）。
    """
    N = yA.size
    D = yAB.shape[0]
    yA = yA - yA.mean()
    yB = yB - yB.mean()
    yAB = yAB - yAB.mean(axis=1, keepdims=True)
    yABr = yABr - yABr.mean(axis=1, keepdims=True)
    V = float((yA ** 2).mean())
    if not np.isfinite(V) or V <= 0.0:
        S2 = (np.full((D, D), np.nan) if yAB2 is not None else None)
        return np.full(D, np.nan), np.full(D, np.nan), S2, V
    S1 = (yA[None, :] * yAB).mean(axis=1) / V
    ST = ((yA[None, :] - yABr) ** 2).mean(axis=1) / (2.0 * V)
    S2 = None
    if yAB2 is not None:
        yAB2 = yAB2 - yAB2.mean(axis=1, keepdims=True)
        S2 = np.zeros((D, D))
        for p, (i, j) in enumerate(pairs):
            S2[i, j] = (yAB2[p] * yA).mean() / V - S1[i] - S1[j]
    return S1, ST, S2, V


def sobol_indices(yA, yB, yAB, yABr, yAB2=None, pairs=None, n_boot=100, rng=None):
    """计算一阶/总效应/二阶 Sobol 指数及 bootstrap 90% CI。

    返回 dict: S1, ST (D,), S2 (D,D) 严格上三角（无 yAB2 时为 None）, V, ci。
    """
    rng = np.random.default_rng(rng)
    D, N = yAB.shape
    if yAB2 is not None and pairs is None:
        raise ValueError("pairs required with yAB2")
    S1, ST, S2, V = _est(yA, yB, yAB, yABr, yAB2, pairs)
    if n_boot:
        bS1 = np.empty((n_boot, D)); bST = np.empty((n_boot, D))
        bS2 = (np.empty((n_boot, D, D)) if yAB2 is not None else None)
        for b in range(n_boot):
            idx = rng.integers(0, N, N)          # 联合按行重采样
            r = _est(yA[idx], yB[idx], yAB[:, idx], yABr[:, idx],
                     None if yAB2 is None else yAB2[:, idx], pairs)
            bS1[b], bST[b] = r[0], r[1]
            if bS2 is not None:
                bS2[b] = r[2]
        ci = {"S1": np.percentile(bS1, [5, 95], axis=0),
              "ST": np.percentile(bST, [5, 95], axis=0),
              "S2": np.percentile(bS2, [5, 95], axis=0) if bS2 is not None else None}
    else:
        ci = None
    # 数值噪声修正：微小负值裁剪到 0
    S1 = np.clip(S1, 0.0, None); ST = np.clip(ST, 0.0, None)
    if S2 is not None:
        S2 = np.clip(S2, 0.0, None)
    if ci and ci["S1"] is not None:
        ci["S1"] = np.clip(ci["S1"], 0.0, None)
        ci["ST"] = np.clip(ci["ST"], 0.0, None)
        if ci["S2"] is not None:
            ci["S2"] = np.clip(ci["S2"], 0.0, None)
    return {"S1": S1, "ST": ST, "S2": S2, "V": V, "ci": ci}


# ------------------------------------------------------------------ 自检
def _ishigami(X):
    a_, b_ = 7.0, 0.1
    return (np.sin(X[:, 0]) + a_ * np.sin(X[:, 1]) ** 2
            + b_ * X[:, 2] ** 4 * np.sin(X[:, 0]))


def self_test(N=8192):
    """Ishigami 自检（含二阶），返回 (ok, 报告文本)。"""
    bounds = np.tile(np.array([-np.pi, np.pi]), (3, 1))
    S = saltelli_sample(bounds, N, rng=2026)
    yA, yB, yAB, yABr, yAB2 = evaluate_matrices(_ishigami, S)
    r = sobol_indices(yA, yB, yAB, yABr, yAB2, S["pairs"], n_boot=0)
    ana_S1 = np.array([0.3139, 0.4424, 0.0])
    ana_ST = np.array([0.5576, 0.4424, 0.2437])
    # Ishigami 的 x3 相关项方差大，二阶容忍放宽
    ok = (np.allclose(r["S1"], ana_S1, atol=0.03)
          and np.allclose(r["ST"], ana_ST, atol=0.03)
          and abs(r["S2"][0, 2] - 0.2437) < 0.06
          and abs(r["S2"][0, 1]) < 0.05 and abs(r["S2"][1, 2]) < 0.05)
    txt = ["[sobol_lib 自检] Ishigami N=%d:" % N,
           "  S1  估计 %s" % np.round(r["S1"], 4),
           "      解析 %s" % ana_S1,
           "  ST  估计 %s" % np.round(r["ST"], 4),
           "      解析 %s" % ana_ST,
           "  S13 = %.4f (解析≈0.2437), S12=%.4f, S23=%.4f (解析=0)"
           % (r["S2"][0, 2], r["S2"][0, 1], r["S2"][1, 2]),
           "  结果: %s" % ("PASS" if ok else "FAIL")]
    return ok, "\n".join(txt)


if __name__ == "__main__":
    ok, txt = self_test()
    print(txt)
    raise SystemExit(0 if ok else 1)
