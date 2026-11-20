# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
"""T21/T22：降阶生产模型标定（Case3，Cn=1，ts=30）。"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

from model.three_phase import ThreePhase
from model.timestepping import advance, ConvergenceError
from model.production import fit_exponential_targets, tank_production

OUT = BASE / "03_结果与图表"
OUT.mkdir(exist_ok=True)

CASE, NF, Q = 3, 10, 300.0
L, N = 40.0, 100
T_INJ = 10.0
T_SOAK = 30.0
DATA = BASE / "01_数据" / "附件C_Synthetic_Production_Data.xlsx"
data = pd.read_excel(DATA)
case_data = data.loc[data["Case_ID"].str.startswith("Case3_")
                     & data["Nano_Concentration_wt%"].eq(1.0)]


def oil_targets(soak_days):
    group = case_data.loc[case_data["Soak_Time_Days"].eq(soak_days)]
    daily = group.set_index("Production_Day")["Oil_Rate_m3_d"]
    cumulative = group.set_index("Production_Day")["Cum_Oil_m3"]
    return float(daily.loc[1]), float(cumulative.loc[360])


m = ThreePhase(L=L, n=N, stretch=0.5, case=CASE, fg_inj=0.5,
               q_inj_ref=Q / 86400.0, N_f=NF)
lines = [f"Case{CASE}, N_f={NF}, q={Q:g}, L={L:g}, n={N}, t_inj={T_INJ:g}",
         "Pressure and saturation are from the three-phase model. Nano concentration labels the data subset.",
         "Oil uses tabulated oil density as the reference baseline. Water reference density is 1000 kg/m3."]
peak = m.prm["p0"]


def record_pressure():
    global peak
    peak = max(peak, m.injection_pressure())
    return peak <= m.prm["p_max"]


def advance_to(target, step_days, callback=None):
    while target - m.t > 1e-12:
        result = advance(m, min(step_days, target - m.t) * 86400.0, tol=1e-9,
                         max_halvings=4, on_step=callback)
        if result is None:
            raise RuntimeError(f"Estimated injection pressure exceeded the limit at {m.t:.3f} d")


area = NF * 4.0 * m.prm["Lf"] * m.prm["hf"]


def predict(length, eta=1.0, cap=100.0, days=360.0):
    return tank_production(m.P, m.Sw, m.Sg, m.prm, area, L, area * length,
                           t_end=days, liquid_cap=cap, eta=eta, cell_widths=m.dx)


try:
    advance_to(T_INJ, 0.1, record_pressure)
    lines.append(f"Injection complete: t={m.t:.2f} d, p_inj_peak_est={peak/1e6:.3f} MPa")
    m.v_inj = 0.0
    advance_to(T_INJ + T_SOAK, 0.2)
    first_day, total = oil_targets(T_SOAK)
    lines.append(f"Calibration data at ts={T_SOAK:g} d: Q1={first_day:.4f} m3, Q360={total:.4f} m3")
    uncapped = predict(L, cap=None)
    full = predict(L)
    qi_o, inventory = fit_exponential_targets(first_day, total)
    if uncapped["q_i_o"] <= 0 or uncapped["V_o"] <= 0:
        raise RuntimeError("Calibration requires positive raw oil rate and available inventory")
    eta = qi_o / uncapped["q_i_o"]
    length = L * inventory / uncapped["V_o"]
    if not (0 < length <= L):
        raise RuntimeError(f"Calibration requests a volume beyond the model domain: L_d={length:g} m")
    fit = predict(length, eta)
    np.testing.assert_allclose([predict(length, eta, days=1.0)["Q_o"], fit["Q_o"]],
                               [first_day, total], rtol=1e-7)
    lines.append(f"Full-domain Q360 at eta=1: {full['Q_o']:.1f} m3")
    lines.append(f"Two-target calibration: productivity eta={eta:.5f}, L_d={length:.5f} m, "
                 f"qi_o={fit['q_i_o']:.3f} m3/d, tau={fit['tau']:.2f} d, Q360={fit['Q_o']:.2f} m3")
    lines.append(f"Initial liquid before rate limit={fit['q_i_liquid_uncapped']:.2f} m3/d. "
                 f"Water proxy Q360={fit['Q_w']:.2f} m3, no water observations in Attachment C.")

    # This is the validation soak level in the planned split. Freeze both coefficients.
    validation_soak = 55
    advance_to(T_INJ + validation_soak, 0.2)
    observed_first, observed_total = oil_targets(validation_soak)
    predicted_first = predict(length, eta, days=1.0)["Q_o"]
    predicted_total = predict(length, eta)["Q_o"]
    lines.append(f"Validation ts={validation_soak} d with frozen eta and L_d: "
                 f"Q1 model/data={predicted_first:.3f}/{observed_first:.3f} m3; "
                 f"Q360 model/data={predicted_total:.2f}/{observed_total:.2f} m3; "
                 f"relative Q360 error={abs(predicted_total-observed_total)/observed_total:.2%}")
except (ConvergenceError, RuntimeError, ValueError, AssertionError) as error:
    raise SystemExit(f"T21/T22 validation failed. No replacement report saved. {error}") from error

lines.append("Scope: Case3 reduced-order calibration and one validation soak. "
             "Case1/2 q=300 baseline exceeds the pressure constraint in T10. "
             "Other inputs, production decline physics, water rates and P3 trends remain unvalidated.")
report = "\n".join(lines) + "\n"
(OUT / "T21_production_calibration.txt").write_text(report, encoding="utf-8")
print(report, end="")
