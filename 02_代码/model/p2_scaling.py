# -*- coding: utf-8 -*-
"""
T08：焖井期压力再分布与改造半径的近似解析解。

模型：常扩散系数 D、半无限域、x=0 零通量（关井）。
把注入末态的压力扰动近似为阶跃 u0(x)=A (0<=x<=a)，用镜像热核：

    u(x,t) = A/2 [ erf((a+x)/(2 sqrt(Dt))) + erf((a-x)/(2 sqrt(Dt))) ]

按题面默认阈值 u(R,t)=0.05 p0 求最远非负根 R(t)。
"""

import numpy as np
from scipy.special import erf


def fit_step_profile(x, dp):
    """由注入末态剖面拟合阶跃幅值 A 与等效宽度 a。"""
    x = np.asarray(x, float)
    dp = np.maximum(np.asarray(dp, float), 0.0)
    A = float(dp[0]) if dp[0] > 0 else float(np.max(dp))
    if A <= 0:
        return 0.0, 0.0
    try:
        area = np.trapezoid(dp, x)
    except AttributeError:          # numpy < 2.0
        area = np.trapz(dp, x)
    a = float(area / A)
    return A, a


def redistribution(x, ts, D, A, a):
    """焖井 ts 天后的压力扰动，Pa。"""
    x = np.asarray(x, float)
    if ts <= 0:
        return np.where((x >= 0) & (x <= a), A, 0.0)
    s = 2.0 * np.sqrt(D * ts * 86400.0)   # ts 以天计，D 为 m2/s
    return 0.5 * A * (erf((a + x) / s) + erf((a - x) / s))


def radius(ts, D, A, a, p0, frac=0.05, ngrid=8000):
    """阈值半径 R(ts)，m；若缝面已低于阈值则返回 0。"""
    thr = frac * p0
    if A <= 0:
        return 0.0
    if redistribution(np.array([0.0]), ts, D, A, a)[0] < thr:
        return 0.0
    xmax = max(4.0 * np.sqrt(D * ts * 86400.0), 2.0 * a, 1.0)
    xg = np.linspace(0.0, xmax, ngrid)
    d = redistribution(xg, ts, D, A, a)
    idx = np.where(d < thr)[0]
    if len(idx) == 0:
        return float(xmax)
    i = int(idx[0])
    if i == 0:
        return 0.0
    x0, x1 = xg[i - 1], xg[i]
    d0, d1 = d[i - 1], d[i]
    return float(x0 + (thr - d0) * (x1 - x0) / (d1 - d0))


def erfc_step(x, t_days, D, p0, pinj):
    """定压阶跃的 erfc 解，仅作校验。"""
    s = 2.0 * np.sqrt(D * t_days * 86400.0)
    return (pinj - p0) * (1.0 - erf(np.asarray(x, float) / s))
