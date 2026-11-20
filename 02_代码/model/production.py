# -*- coding: utf-8 -*-
"""
T21/T22：降阶生产模型（物质平衡 + 指数递减）。

输入是焖井末态的 p(x)、S_w(x)、S_g(x)。假设生产期饱和度分布缓变，
用平均流度给出初始产率，用可动油体积给出递减时间：

    q_i = lambda_o * (p_avg - p_wf) / L * A
    (q_o + q_w) <= 日产液上限
    V_o = V_bulk * phi * (So_avg - Sor)
    tau = V_o / q_i
    Q_o = V_o * (1 - exp(-t_end/tau))

绝对量级由附件 C 标定；相对趋势用于 P3。
"""

import numpy as np


def tank_production(p, Sw, Sg, prm, area, L, V_bulk,
                    t_end=360.0, liquid_cap=100.0, eta=1.0):
    krw = prm["krw0"] * np.clip((Sw - prm["Swr"]) /
                                (1 - prm["Swr"] - prm["Sor"]), 1e-6, 1) ** prm["nw"]
    Sod = np.clip(1 - (Sw - prm["Swr"]) / (1 - prm["Swr"] - prm["Sor"])
                  - (Sg - prm["Sgc"]) / (1 - prm["Swr"] - prm["Sor"]), 1e-6, 1)
    kro = prm["kro0"] * Sod ** prm["no"]

    lam_o = float(np.mean(prm["k"] * kro / prm["mu_o"]))
    lam_w = float(np.mean(prm["k"] * krw / prm["mu_w"]))
    p_avg = float(np.mean(p))
    dp = max(p_avg - prm["p_wf"], 0.0)

    qi_o = lam_o * dp / L * area * 86400.0
    qi_w = lam_w * dp / L * area * 86400.0

    qt = qi_o + qi_w
    if liquid_cap is not None and qt > liquid_cap and qt > 0:
        s = liquid_cap / qt
        qi_o *= s
        qi_w *= s

    phi_avg = float(np.mean(prm["phi0"] * (1 + prm["cphi"] * (p - prm["p0"]))))
    So_avg = float(np.mean(np.clip(1 - Sw - Sg, 0.0, 1.0)))
    movable = max(So_avg - prm["Sor"], 0.0)
    V_o = V_bulk * phi_avg * movable          # 可动油体积，m3

    if qi_o <= 0 or V_o <= 0:
        return {"q_i_o": 0.0, "q_i_w": 0.0, "Q_o": 0.0, "Q_w": 0.0,
                "tau": np.inf, "V_o": V_o, "eta": eta, "p_avg": p_avg}

    tau = V_o / qi_o
    decay = 1.0 - np.exp(-t_end / tau)
    Q_o = eta * V_o * decay
    Q_w = qi_w * tau * decay
    return {"q_i_o": qi_o, "q_i_w": qi_w, "Q_o": Q_o, "Q_w": Q_w,
            "tau": tau, "V_o": V_o, "eta": eta, "p_avg": p_avg}
