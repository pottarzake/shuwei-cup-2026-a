# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
"""
economics.py —— 附录 B 经济评价模块（队友负责，2026-11-20，T16/T17）

口径铁律（错一处即违规，修改前读 00_题目与规范/A题要点整理.md §5 与原文附录 B）：
  1. 参考状态 25 MPa、300 K；ρ_CO2,ref=850 kg/m³，ρ_aq=1000 kg/m³
  2. C_n 以 wt% 原值代入，m_nano = (C_n/100)·ρ_aq·V_aq/1000（t）
  3. 纳米上限 30 t/井次，超过即不可行
  4. t_inj = V_inj/q_inj；候选方案 T = τ_inj + t_inj + t_s + 360 d；
     B1 基准（仅压裂）T = 360 d
  5. 逐日折现：NPV = -C_frac + Σ_t CF(t)/(1+r)^(t/365)，折现基准第 0 天，日终
  6. 注入阶段按实际注入天数计 c_inj；焖井计 c_soak；生产阶段只计
     可变生产成本 c_prod·q_o 与采出水处理费 c_water·q_w（无日作业费）
  7. 单位统一 10^4 CNY（万元）

数据缺口（总纲 Q3）：附件 C 无产水数据。Q_w 用 WOR 系数假设：
   q_w(t) = wor · q_o(t)，默认 wor=0.5，敏感性见 T17 输出。

运行自检：python economics.py  → 必须复现题面算例 848.25 万元
"""

import numpy as np

# ---------------- B.1 经济参数（万元口径） ----------------
P_OIL_STAR = 0.3396      # 万元/m³（75 USD/bbl × 7.2 / 0.159 ≈ 3396 CNY/m³）
C_NANO = 20.0            # 万元/t
C_CO2 = 0.03             # 万元/t
C_INJ = 30.0             # 万元/d
C_SOAK = 3.0             # 万元/d
C_FRAC = 120.0           # 万元/井次
C_PROD = 0.035           # 万元/m³ 油
C_WATER = 0.0025         # 万元/m³ 水
R_DISC = 0.08            # 年折现率，逐日 (1+r)^(t/365)
RHO_CO2_REF = 850.0      # kg/m³
RHO_AQ = 1000.0          # kg/m³
NANO_LIMIT_T = 30.0      # t/井次


def materials(V_inj, f_co2, cn):
    """物料用量与费用。V_inj(m³), f_co2(-), cn(wt% 原值)。
    返回 dict；nano_ok=False 表示超过 30 t 上限（判不可行）。"""
    V_co2 = f_co2 * V_inj
    V_aq = (1.0 - f_co2) * V_inj
    m_co2 = RHO_CO2_REF * V_co2 / 1000.0          # t
    m_nano = (cn / 100.0) * RHO_AQ * V_aq / 1000.0  # t
    return {"V_co2": V_co2, "V_aq": V_aq,
            "m_co2": m_co2, "m_nano": m_nano,
            "C_co2": C_CO2 * m_co2, "C_nano": C_NANO * m_nano,
            "nano_ok": m_nano <= NANO_LIMIT_T + 1e-12}


def operation_cost(V_inj, q_inj, f_co2, cn, t_s, tau_inj=0.0):
    """C_total 中的非生产部分（压裂+注入+焖井+材料），即题面 848.25 的口径。
    注意：C_total 全口径另含 c_prod·Q_o + c_water·Q_w，见 candidate_npv。"""
    m = materials(V_inj, f_co2, cn)
    t_inj = V_inj / q_inj
    C_op = (C_FRAC + C_INJ * t_inj + C_SOAK * t_s
            + m["C_nano"] + m["C_co2"])
    return {"C_op": C_op, "t_inj": t_inj, **m}


def _discount(days):
    """days: 数组（从第 0 天起算的日终时刻）→ 折现因子 (1+r)^(-t/365)"""
    return (1.0 + R_DISC) ** (-np.asarray(days, float) / 365.0)


def _phase_daily_cf(start, duration, daily_amount):
    """在 [start, start+duration) 上按 1 d 步长铺日流量，边界日按占比折算。
    返回 (days, amounts) 两个数组。"""
    if duration <= 0:
        return np.array([]), np.array([])
    n_full = int(np.floor(duration + 1e-12))
    frac = duration - n_full
    days = start + np.arange(1, n_full + 1, dtype=float)
    amts = np.full(n_full, daily_amount, dtype=float)
    if frac > 1e-9:
        days = np.append(days, start + n_full + frac)  # 期末时点
        amts = np.append(amts, daily_amount * frac)
    return days, amts


def candidate_npv(V_inj, q_inj, f_co2, cn, t_s, tau_inj,
                  oil_daily, water_daily=None, wor=0.5,
                  p_oil_star=P_OIL_STAR):
    """候选方案 NPV（万元）。oil_daily: 360 天日产油数组。
    t_inj = V_inj/q_inj 可为分数；τ_inj 只延迟折现，不产生费用。"""
    m = materials(V_inj, f_co2, cn)
    t_inj = V_inj / q_inj
    if water_daily is None:
        water_daily = wor * np.asarray(oil_daily, float)

    days_all, cf_all = [], []
    # 注入阶段：日费 + 材料费按注入天数均摊
    d, a = _phase_daily_cf(tau_inj, t_inj,
                           -(C_INJ + (m["C_nano"] + m["C_co2"]) / t_inj))
    days_all.append(d); cf_all.append(a)
    # 焖井阶段
    d, a = _phase_daily_cf(tau_inj + t_inj, t_s, -C_SOAK)
    days_all.append(d); cf_all.append(a)
    # 生产阶段（360 d，index 与 oil_daily 对齐）
    start = tau_inj + t_inj + t_s
    cf_prod = p_oil_star * np.asarray(oil_daily, float) \
        - C_PROD * np.asarray(oil_daily, float) \
        - C_WATER * np.asarray(water_daily, float)
    days_all.append(start + np.arange(1, 361, dtype=float))
    cf_all.append(cf_prod)

    days = np.concatenate(days_all)
    cf = np.concatenate(cf_all)
    npv = -C_FRAC + float(np.sum(cf * _discount(days)))
    return {"NPV": npv, "t_inj": t_inj, "nano_ok": m["nano_ok"],
            "m_nano": m["m_nano"], "m_co2": m["m_co2"],
            "Q_o": float(np.sum(oil_daily)), "Q_w": float(np.sum(water_daily))}


def b1_npv(oil_daily, water_daily=None, wor=0.5, p_oil_star=P_OIL_STAR):
    """B1 基准（仅压裂，不注复合流体）：T=360 d，生产从第 1 天起。"""
    oil_daily = np.asarray(oil_daily, float)
    if water_daily is None:
        water_daily = wor * oil_daily
    cf = (p_oil_star - C_PROD) * oil_daily - C_WATER * np.asarray(water_daily, float)
    days = np.arange(1, 361, dtype=float)
    return -C_FRAC + float(np.sum(cf * _discount(days)))


def breakeven_oil_price(V_inj, q_inj, f_co2, cn, t_s, tau_inj,
                        oil_daily, water_daily=None, wor=0.5, npv_target=0.0):
    """求使 NPV = npv_target 的油价 P（万元/m³），二分法。
    NPV(P) = -C_FRAC + 注入焖井折现 + Σ[(P-c_prod)q_o - c_water q_w]·DF，对 P 单调。"""
    m = materials(V_inj, f_co2, cn)
    t_inj = V_inj / q_inj
    oil = np.asarray(oil_daily, float)
    water = wor * oil if water_daily is None else np.asarray(water_daily, float)

    def npv_at(p):
        days_all, cf_all = [], []
        d, a = _phase_daily_cf(tau_inj, t_inj,
                               -(C_INJ + (m["C_nano"] + m["C_co2"]) / t_inj))
        days_all.append(d); cf_all.append(a)
        d, a = _phase_daily_cf(tau_inj + t_inj, t_s, -C_SOAK)
        days_all.append(d); cf_all.append(a)
        start = tau_inj + t_inj + t_s
        cf_prod = (p - C_PROD) * oil - C_WATER * water
        days_all.append(start + np.arange(1, 361, dtype=float))
        cf_all.append(cf_prod)
        return -C_FRAC + float(np.sum(np.concatenate(cf_all)
                                      * _discount(np.concatenate(days_all))))

    lo, hi = 0.0, 5.0   # 0 ~ 5 万元/m³  bracket
    if npv_at(lo) > npv_target:
        return 0.0        # 即使油价为 0 也达标
    if npv_at(hi) < npv_target:
        return np.nan     # 5 万元/m³ 内无法盈亏平衡
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if npv_at(mid) < npv_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def selfcheck_848_25():
    """题面算例自检：V_inj=3000, f=0.5, Cn=1.0, q_inj=300, t_s=30
    → 纳米 15 t/300 万，CO₂ 1275 t/38.25 万，注入 10 d/300 万，
      焖井 30 d/90 万，压裂 120 万，小计 848.25 万。"""
    r = operation_cost(V_inj=3000, q_inj=300, f_co2=0.5, cn=1.0, t_s=30)
    parts = {"C_nano": 300.0, "C_co2": 38.25, "C_inj": 300.0,
             "C_soak": 90.0, "C_frac": 120.0}
    ok = abs(r["C_op"] - 848.25) < 1e-6
    detail = {k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}
    return ok, r["C_op"], parts, detail


if __name__ == "__main__":
    ok, total, parts, detail = selfcheck_848_25()
    print(f"[T16 self-check] 题面算例 848.25 万元, 复算 = {total:.6f} 万元, "
          f"{'PASS' if ok else 'FAIL'}")
    for k, v in parts.items():
        print(f"  {k}: {v}")
    print(f"  detail: {detail}")
    assert ok, "自检不过，禁止用于论文！"
