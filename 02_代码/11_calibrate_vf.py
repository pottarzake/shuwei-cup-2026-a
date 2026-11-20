# -*- coding: utf-8 -*-
"""T31: local V/f sensitivity of the Case3 injection–soak–tank proxy."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

from model.p3_forward import _row, _sat
from model.production import tank_production
from model.three_phase import ThreePhase
from model.timestepping import advance, ConvergenceError

OUT = BASE / "03_结果与图表"
CASE, NF, Q, L, N, T_SOAK = 3, 10, 300.0, 40.0, 100, 30.0
GRID = [(1000, 0.2), (1000, 0.8), (3000, 0.2), (3000, 0.5),
        (3000, 0.8), (5000, 0.2), (5000, 0.8)]


def evaluate_point(pair):
    volume, fraction = pair
    m = ThreePhase(L=L, n=N, stretch=0.5, case=CASE, fg_inj=fraction,
                   q_inj_ref=Q / 86400.0, N_f=NF)
    cap = m.prm["p_max"]
    nominal_end = volume / Q
    peak = m.prm["p0"]

    def pressure_ok():
        nonlocal peak
        peak = max(peak, m.injection_pressure())
        return peak <= cap

    status = "ok"
    try:
        while nominal_end - m.t > 1e-12:
            result = advance(m, min(0.1, nominal_end - m.t) * 86400,
                             tol=1e-9, max_halvings=4, on_step=pressure_ok)
            if result is None:
                status = "overpressure"
                break
        injection_stop = m.t
        if status == "ok":
            m.v_inj = 0.0
            end = m.t + T_SOAK
            while end - m.t > 1e-12:
                advance(m, min(0.2, end - m.t) * 86400, tol=1e-9, max_halvings=4)
    except ConvergenceError as error:
        injection_stop = min(m.t, nominal_end)
        status = f"numerical_failure at {m.t:g} d: {error}"

    if status == "ok":
        fit = _row(CASE, 1.0)
        length = _sat(T_SOAK, fit["Ld_a"], fit["Ld_b"], fit["Ld_c"])
        area = NF * 4.0 * m.prm["Lf"] * m.prm["hf"]
        tank = tank_production(m.P, m.Sw, m.Sg, m.prm, area, L, area * length,
                               cell_widths=m.dx)
        values = [tank[key] for key in ("Q_o", "q_i_o", "tau", "p_avg", "V_o")]
    else:
        values = [np.nan] * 5
    return [volume, fraction, nominal_end, status, *values, injection_stop, peak / 1e6]


def report_fit(df):
    usable = df[(df.status == "ok") & (df.Q360 > 0)]
    baseline = usable[(usable.V == 3000) & np.isclose(usable.f, 0.5)]
    if len(baseline) != 1 or len(usable) < 3:
        raise ValueError("Insufficient feasible positive-Q points or missing baseline")
    q0 = baseline.Q360.item()
    x = np.column_stack([np.log(usable.V / 3000), np.log(usable.f / 0.5)])
    y = np.log(usable.Q360 / q0)
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < 2:
        raise ValueError("V/f coefficients cannot be separated in this sample")
    residual = y - x @ coef
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(residual**2) / ss_tot if ss_tot > 0 else np.nan
    lines = [f"Anchored descriptive fit: a={coef[0]:+.5f}, b={coef[1]:+.5f}, R2={r2:.3f}, n={len(usable)}.",
             f"Q0={q0:.2f} m3. R2 describes these proxy outputs."]
    lines.append("Matched comparisons of f=0.8 vs f=0.2 at fixed V:")
    for volume, group in usable.groupby("V"):
        paired = group.set_index("f").Q360
        if 0.2 in paired and 0.8 in paired:
            gain = paired.loc[0.8] / paired.loc[0.2] - 1
            lines.append(f"  V={volume:g}: Q shift={gain:+.3%}")
    return lines


def main():
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(evaluate_point, GRID))
    columns = ["V", "f", "t_inj", "status", "Q360", "qi", "tau", "p_avg",
               "V_o", "injection_stop_d", "peak_MPa"]
    df = pd.DataFrame(rows, columns=columns)
    lines = ["Case3, N_f=10, q=300 m3/d, ts=30 d, 100 cells.",
             "Fixed calibrated storage length; three-phase flow treats CO2 as insoluble.",
             "Cn=1 locates the T22 fit. Nano feedback is absent from these flow runs.",
             df.round(4).to_string(index=False), "", *report_fit(df),
             "P3 uses zero exponents as a conservative modeling convention. "
             "These runs do not identify effects across other cases, flow rates or soak times."]
    df.to_csv(OUT / "T31_vf_calibration.csv", index=False, encoding="utf-8-sig")
    report = "\n".join(lines) + "\n"
    (OUT / "T31_vf_calibration.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
