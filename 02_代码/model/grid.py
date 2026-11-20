# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
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
    if not np.isfinite(L) or L <= 0 or n < 1:
        raise ValueError("L must be positive and finite, and n must be at least 1")
    if first is None:
        first = stretch * L / n
    first = float(first)
    if not np.isfinite(first) or first <= 0:
        raise ValueError("first cell width must be positive and finite")

    if n == 1 or first >= L / n:
        dx = np.full(n, L / n)
    else:
        # 在对数空间求公比。上界对应末层厚度 L，避免直接计算 r**n。
        indices = np.arange(n)
        log_first_fraction = np.log(first) - np.log(L)
        lo, hi = 0.0, -log_first_fraction / (n - 1)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.exp(log_first_fraction + mid * indices).sum() < 1.0:
                lo = mid
            else:
                hi = mid
        dx_fraction = np.exp(log_first_fraction + 0.5 * (lo + hi) * indices)
        dx = L * (dx_fraction / dx_fraction.sum())

    xf = np.concatenate(([0.0], np.cumsum(dx)))
    xc = 0.5 * (xf[:-1] + xf[1:])
    return xf, xc, dx
