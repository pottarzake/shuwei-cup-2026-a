# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
"""
T18/T19：一维孔隙弹性增量本构与压缩损伤。

约定：应变拉伸为正，正应力压缩为正。
总应力沿 x 恒定：sigma_xx = sigma_h。
有效应力：sigma'_xx = sigma_h - alpha p。
相对初始状态：
    Delta sigma' = -alpha (p - p0)
    eps = (sigma'_0 - sigma') / M = alpha (p - p0) / M
其中 M 为一维应变（uniaxial strain）约束模量：
    M = E (1-nu) / ((1+nu)(1-2nu))

损伤由压缩应变历史驱动：
    kappa = max_{t'<=t} <-eps(t')>_+
    d = 1 - exp[-beta_d <kappa - eps_d>_+]
"""

import numpy as np


def constrained_modulus(E, nu):
    return E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))


def uniaxial_strain(p, p0, alpha, M):
    """拉伸为正的 x 方向应变。"""
    return alpha * (np.asarray(p, float) - p0) / M


def damage_from_strain_history(eps, eps_d, beta_d):
    """
    eps: 时间序列（标量列表或数组）。
    返回 (kappa, d) 序列。
    """
    eps = np.asarray(eps, float)
    kappa = np.maximum.accumulate(np.maximum(-eps, 0.0))
    d = 1.0 - np.exp(-beta_d * np.maximum(kappa - eps_d, 0.0))
    return kappa, d


def property_update(phi0, k0, p, p0, cphi, gamma_k, d,
                    lambda_d=0.5, xi=3.0):
    """损伤与压敏对孔隙度、基质渗透率的折减。"""
    phi = phi0 * (1.0 + cphi * (p - p0)) * (1.0 - lambda_d * d)
    k = k0 * np.exp(gamma_k * (p - p0)) * (1.0 - d) ** xi
    return phi, k


def case_mechanics(case=3):
    """从 config 取参数并返回 M、注入/生产应变、名义损伤。"""
    import config
    c = config.case_params(case)
    M = constrained_modulus(c["E"], c["ν"])
    eps_inj = uniaxial_strain(c["p_max"], c["p_0"], c["α"], M)
    eps_prod = uniaxial_strain(c["p_wf"], c["p_0"], c["α"], M)
    kappa = max(-eps_inj, -eps_prod, 0.0)
    d = 1.0 - np.exp(-c["β_d"] * max(kappa - c["ε_d"], 0.0))
    return {
        "M": M, "eps_inj": eps_inj, "eps_prod": eps_prod,
        "kappa": kappa, "d": d, "eps_d": c["ε_d"], "beta_d": c["β_d"],
    }
