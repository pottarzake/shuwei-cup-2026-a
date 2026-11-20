# -*- coding: utf-8 -*-
"""一维裂缝面法向网格：近缝面几何加密。

默认按相对加密 first = stretch * L / n，保证网格整体加密时
最小单元同比例缩小，空间收敛阶不会被固定的首层厚度卡住。
"""

import numpy as np


def make_grid(L, n=400, stretch=0.3, first=None):
    """
    生成 x∈[0, L] 的一维有限体积网格。

    参数
    ----
    L : 域长
    n : 单元数
    stretch : 相对首层厚度系数，first = stretch * L / n；1.0 表示均匀
    first : 若给出，则直接指定首层厚度（绝对长度），忽略 stretch

    返回
    ----
    xf, xc, dx
    """
    L = float(L)
    n = int(n)
    if first is None:
        first = stretch * L / n
    first = float(first)

    if abs(first * n - L) < 1e-12 or first * n >= L:
        dx = np.full(n, L / n)
    else:
        def total(r):
            return first * (r ** n - 1.0) / (r - 1.0)

        lo, hi = 1.0 + 1e-12, 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if total(mid) < L:
                lo = mid
            else:
                hi = mid
        r = 0.5 * (lo + hi)
        dx = first * r ** np.arange(n)
        dx *= L / dx.sum()

    xf = np.concatenate(([0.0], np.cumsum(dx)))
    xc = 0.5 * (xf[:-1] + xf[1:])
    return xf, xc, dx
