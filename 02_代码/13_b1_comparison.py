# OpenAI Codex (GPT-6 family, exact release date unconfirmed) assisted review or edits on 2026-11-20.
"""Fracturing-only counterfactual and Appendix B incremental NPV."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

import economics
from model.production import tank_production
from model.three_phase import build_prm

OUT = BASE / "03_结果与图表"
WOR = 0.5
LENGTH = 40.0
FRACTURES = 10


def baseline(case, storage_length):
    """No injection, initial reservoir state, and the shared tank proxy."""
    prm = build_prm(case=case, N_f=FRACTURES)
    area = FRACTURES * 4 * prm["Lf"] * prm["hf"]
    tank = tank_production(np.array([prm["p0"]]), np.array([prm["Sw0"]]),
                           np.zeros(1), prm, area, LENGTH, area * storage_length,
                           liquid_cap=None)
    rate = min(tank["q_i_o"], 100.0 / (1 + WOR))
    if rate > 0 and tank["V_o"] > 0:
        tau = tank["V_o"] / rate
        oil = -tank["V_o"] * np.expm1(-1 / tau) * np.exp(-np.arange(360) / tau)
    else:
        oil = np.zeros(360)
    return float(oil.sum()), economics.b1_npv(oil, WOR * oil)


def main():
    fits = pd.read_csv(OUT / "T22_Ld_parameters.csv")
    optima = pd.read_csv(OUT / "T17_optimal_ts.csv")
    assert economics.selfcheck_848_25()[0]
    rows = []
    for case in (1, 2, 3):
        length = fits.loc[fits.case.eq(case) & fits.Cn.eq(1), "Ld_30"].item()
        total_oil, npv_b1 = baseline(case, length)
        no_storage_npv = baseline(case, 0.0)[1]
        label = f"Case{case}"
        for row in optima.loc[optima.case.eq(label)].itertuples():
            zero_delta_length = np.nan
            if no_storage_npv <= row.NPV_max <= npv_b1:
                zero_delta_length = brentq(
                    lambda storage: baseline(case, storage)[1] - row.NPV_max,
                    0.0, length,
                )
            rows.append((label, row.cn, row.ts_star, length, total_oil,
                         row.NPV_max, npv_b1, row.NPV_max - npv_b1, zero_delta_length))
    report = pd.DataFrame(rows, columns=["case", "cn", "ts_star", "B1_Ld_m",
        "B1_Q360_m3", "candidate_NPV_10kCNY", "B1_NPV_10kCNY", "delta_NPV_10kCNY",
        "B1_zero_delta_Ld_m"])
    report.to_csv(OUT / "T17_b1_comparison.csv", index=False, float_format="%.6f")
    print(report.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
