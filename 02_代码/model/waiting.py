# -*- coding: utf-8 -*-
"""
T23：注入等待期（τ_inj）的裂缝闭合模型。

压裂结束后缝内压力从破裂压力 p_f 衰减到 p0：
    p_f(τ) = p0 + (p_f - p0) exp(-τ/τ_c)
缝面有效正应力：
    σ'_n(τ) = σ_h - α p_f(τ)
缝宽随有效正应力非线性衰减（P1-07 / P1-11）：
    w(σ') = w_inf + (w0 - w_inf) exp[-c_c (σ' - σ'_0)]
平行板立方定律给出导流能力折减：
    k_f = w^2/12,  k_f w = w^3/12  ->  multiplier = (w/w0)^3
"""

import numpy as np

import config


def state(tau, case, tau_c=2.0, c_c=5e-8, w_inf_ratio=0.2):
    """tau: 等待天数；c_c: MPa^-1 换算为 Pa^-1 后传入（默认 0.05 MPa^-1）。"""
    c = config.case_params(case)
    p0, pf, sh, alpha = c["p_0"], c["p_f"], c["σ_h"], c["α"]
    w0 = c["w_f"]
    p = p0 + (pf - p0) * np.exp(-tau / tau_c)
    sn0 = sh - alpha * pf
    sn = sh - alpha * p
    winf = w_inf_ratio * w0
    w = winf + (w0 - winf) * np.exp(-c_c * (sn - sn0))
    mult = (w / w0) ** 3
    return {"tau": tau, "p_frac": p, "sigma_n": sn, "w": w,
            "w0": w0, "mult": mult}
