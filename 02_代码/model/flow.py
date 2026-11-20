# -*- coding: utf-8 -*-
"""
单相压力扩散求解器（T06）。

    d p / d t = D * d2p / dx2

定压边界：p(0,t)=pL, p(L,t)=pR；初始 p(x,0)=p0。
有限体积 + 向后 Euler + 三对角求解。
"""

from math import isfinite

import numpy as np
from scipy.linalg import solve_banded

from .grid import make_grid


def solve_diffusion(L, D, p0, pL, pR, t_end, dt, n=400, first=None,
                    stretch=0.3, store=False):
    """
    返回 (xc, p, history)
    history 仅在 store=True 时给出 [(t, p_array), ...]
    """
    if not isfinite(D) or D <= 0:
        raise ValueError("D must be positive and finite")
    if not isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if not isfinite(t_end) or t_end < 0:
        raise ValueError("t_end must be nonnegative and finite")
    _, xc, dx = make_grid(L, n, stretch=stretch, first=first)
    N = len(dx)

    # 界面距离
    dx_face = 0.5 * (dx[:-1] + dx[1:])          # 内部界面，长度 N-1

    p = np.full(N, float(p0))
    hist = []
    operator = np.zeros((3, N))
    left_rates = D / (dx[1:] * dx_face)
    right_rates = D / (dx[:-1] * dx_face)
    operator[0, 1:] = -right_rates
    operator[2, :-1] = -left_rates
    operator[1, 1:] = left_rates
    operator[1, :-1] += right_rates
    boundary_left = 2.0 * D / dx[0]**2
    boundary_right = 2.0 * D / dx[-1]**2
    operator[1, 0] += boundary_left
    operator[1, -1] += boundary_right
    forcing = np.zeros(N)
    forcing[0] += boundary_left * pL
    forcing[-1] += boundary_right * pR

    def step_matrix(step_size):
        matrix = operator * step_size
        matrix[1] += 1.0
        return matrix

    full_matrix = step_matrix(dt)
    elapsed = 0.0
    step = 0
    while elapsed < t_end:
        step += 1
        target = min(step * dt, t_end)
        if target == step * dt:
            step_size = dt
            matrix = full_matrix
        else:
            step_size = target - elapsed
            matrix = step_matrix(step_size)

        rhs = p + step_size * forcing
        if N == 1:
            p = rhs / matrix[1]
        else:
            p = solve_banded((1, 1), matrix, rhs)
        elapsed = target
        if store:
            hist.append((elapsed, p.copy()))

    return xc, p, hist


def erfc_solution(x, t, D, p0, pL):
    """半无限域、定压阶跃的解析解。"""
    from scipy.special import erfc
    return p0 + (pL - p0) * erfc(x / (2.0 * np.sqrt(D * t)))
