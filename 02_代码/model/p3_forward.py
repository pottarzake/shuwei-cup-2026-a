# -*- coding: utf-8 -*-
"""
P3 快速评估器：数据标定的焖井/浓度维度 + 机理/假设的 V、f 缩放。

基准产量来自 T22 的 Q1(ts)、L_d(ts) 拟合（附件 C 标定）。
V、f 的指数是待验证假设，默认 0.2 / 0.1，可调并在论文中说明。
"""

from pathlib import Path
import numpy as np
import pandas as pd

import economics

BASE = Path(__file__).resolve().parents[2]
_TAB = pd.read_csv(BASE / "03_结果与图表" / "T22_Ld_parameters.csv")

# case -> (area, phi, movable, q_max from T10 with N_f=10)
CASE = {
    1: (80000.0, 0.08, 0.58, 14.6),
    2: (96000.0, 0.10, 0.55, 137.0),
    3: (112000.0, 0.13, 0.52, 1510.0),
}
BOUNDS = {
    "cn": (0.5, 1.5), "q": (100.0, 800.0), "V": (1000.0, 5000.0),
    "f": (0.2, 0.8), "ts": (5.0, 60.0), "tau": (0.0, 15.0),
}
NAMES = ["cn", "q", "V", "f", "ts", "tau"]


def _row(case, cn):
    sub = _TAB[_TAB.case == case].sort_values("Cn")
    if sub.empty:
        raise ValueError(f"no fit row for case={case}")
    cn = float(np.clip(cn, sub.Cn.min(), sub.Cn.max()))
    out = {}
    for col in ("Ld_a", "Ld_b", "Ld_c", "q_a", "q_b", "q_c"):
        out[col] = float(np.interp(cn, sub.Cn, sub[col]))
    return pd.Series(out)


def _sat(t, a, b, c):
    return a + b * (1.0 - np.exp(-t / c))


def evaluate(case, cn, q, V, f, ts, tau, exp_v=0.2, exp_f=0.1, wor=0.5,
             k_r=1.0, phi_r=1.0, sh_r=1.0):
    area, phi, movable, qmax0 = CASE[case]
    qmax = qmax0 * k_r / (sh_r ** 0.5)
    phi = phi * phi_r
    r = _row(case, cn)

    # 可行性
    viol = 0.0
    if q > qmax:
        viol += (q - qmax) / qmax
    t_inj = V / q
    if t_inj + ts > 90.0:
        viol += (t_inj + ts - 90.0) / 90.0
    m = economics.materials(V, f, cn)
    if not m["nano_ok"]:
        viol += (m["m_nano"] - economics.NANO_LIMIT_T) / economics.NANO_LIMIT_T

    # 数据标定的基准 + V/f 缩放
    sv = (V / 3000.0) ** exp_v
    sf = (f / 0.5) ** exp_f
    q1 = _sat(ts, r.q_a, r.q_b, r.q_c) * sv * sf * (k_r ** 0.5)
    Ld = _sat(ts, r.Ld_a, r.Ld_b, r.Ld_c) * sv * sf
    V_o = area * Ld * phi * movable

    qw1 = wor * q1
    if q1 + qw1 > 100.0:
        s = 100.0 / (q1 + qw1)
        q1 *= s
        qw1 *= s
    tau_dec = V_o / q1 if q1 > 0 else np.inf
    t = np.arange(1, 361, dtype=float)
    oil = q1 * np.exp(-t / tau_dec)
    water = wor * oil

    res = economics.candidate_npv(V, q, f, cn, ts, tau, oil, water)
    C_total = (economics.operation_cost(V, q, f, cn, ts, tau)["C_op"]
               + economics.C_PROD * oil.sum()
               + economics.C_WATER * water.sum())
    return {
        "feasible": viol <= 1e-12,
        "violation": float(viol),
        "Q_o": float(oil.sum()),
        "Q_w": float(water.sum()),
        "C_total": float(C_total),
        "NPV": float(res["NPV"]),
        "q1": float(q1),
        "Ld": float(Ld),
        "t_inj": float(t_inj),
        "m_nano": float(m["m_nano"]),
    }


def decode(x):
    lo = np.array([BOUNDS[n][0] for n in NAMES])
    hi = np.array([BOUNDS[n][1] for n in NAMES])
    return lo + np.clip(x, 0, 1) * (hi - lo)
