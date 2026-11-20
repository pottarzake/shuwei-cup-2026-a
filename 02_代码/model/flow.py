# -*- coding: utf-8 -*-
"""
单相压力扩散求解器（T06）。

    d p / d t = D * d2p / dx2

定压边界：p(0,t)=pL, p(L,t)=pR；初始 p(x,0)=p0。
有限体积 + 向后 Euler + 三对角求解。
"""

import numpy as np
from scipy.linalg import solve_banded

from .grid import make_grid


def solve_diffusion(L, D, p0, pL, pR, t_end, dt, n=400, first=None,
                    stretch=0.3, store=False):
    """
    返回 (xc, p, history)
    history 仅在 store=True 时给出 [(t, p_array), ...]
    """
    xf, xc, dx = make_grid(L, n, stretch=stretch, first=first)
    N = len(dx)

    # 界面距离
    dx_face = 0.5 * (dx[:-1] + dx[1:])          # 内部界面，长度 N-1

    p = np.full(N, float(p0))
    nstep = int(round(t_end / dt))
    hist = []

    for step in range(1, nstep + 1):
        # 三对角系数：-a_i p_{i-1} + b_i p_i - c_i p_{i+1} = rhs_i
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        rhs = p.copy()

        # 内部单元 i=1..N-2
        a[1:N-1] = D * dt / (dx[1:N-1] * dx_face[0:N-2])
        c[1:N-1] = D * dt / (dx[1:N-1] * dx_face[1:N-1])
        b[1:N-1] = 1.0 + a[1:N-1] + c[1:N-1]

        # 左边界：x=0 定压，距首单元中心 dx0/2
        a[0] = 0.0
        c[0] = D * dt / (dx[0] * dx_face[0])              # 到单元 1
        b_left = D * dt / (dx[0] * (0.5 * dx[0]))         # 到边界
        b[0] = 1.0 + c[0] + b_left
        rhs[0] += b_left * pL

        # 右边界：x=L 定压
        c[N-1] = 0.0
        a[N-1] = D * dt / (dx[N-1] * dx_face[-1])         # 到单元 N-2
        b_right = D * dt / (dx[N-1] * (0.5 * dx[N-1]))    # 到边界
        b[N-1] = 1.0 + a[N-1] + b_right
        rhs[N-1] += b_right * pR

        # 组装带状矩阵（上对角、主对角、下对角）
        ab = np.zeros((3, N))
        ab[0, 1:] = -c[:-1]     # 主对角上方
        ab[1, :] = b
        ab[2, :-1] = -a[1:]     # 主对角下方

        p = solve_banded((1, 1), ab, rhs)
        if store:
            hist.append((step * dt, p.copy()))

    return xc, p, hist


def erfc_solution(x, t, D, p0, pL):
    """半无限域、定压阶跃的解析解。"""
    from scipy.special import erfc
    return p0 + (pL - p0) * erfc(x / (2.0 * np.sqrt(D * t)))
