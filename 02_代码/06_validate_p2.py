# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
"""T08: compare the three-phase shut-in calculation with two diffusion models."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

import config
from model.p2_scaling import (
    finite_domain_solution, fit_step_profile, radius, redistribution, threshold_radius,
)
from model.three_phase import ThreePhase
from model.timestepping import ConvergenceError, advance

OUT = BASE / "03_结果与图表"
CASE, FRACTURES, LENGTH = 3, 10, 40.0
RATES = (300, 800)


def calculate(rate, cells, refined):
    dt_inj, dt_soak = (0.05, 0.1) if refined else (0.1, 0.2)
    targets = (1, 2, 3, 4, 5, 6, 8, 10)
    if not refined:
        targets += (20, 30, 45, 60)

    m = ThreePhase(L=LENGTH, n=cells, stretch=0.5, case=CASE, fg_inj=0.5,
                   q_inj_ref=rate / 86400.0, N_f=FRACTURES)
    p0, D = m.prm["p0"], config.hydraulic_diffusivity(CASE)
    threshold = 0.05 * p0
    peak = p0

    def record_pressure():
        nonlocal peak
        peak = max(peak, m.injection_pressure())
        return peak <= m.prm["p_max"]

    duration = 3000.0 / rate
    while duration - m.t > 1e-12:
        result = advance(m, min(dt_inj, duration - m.t) * 86400.0,
                         tol=1e-9, max_halvings=4, on_step=record_pressure)
        if result is None:
            raise RuntimeError(f"q={rate}: injection pressure exceeded at {m.t:.4f} d")
    print(f"q={rate}: injection complete at {m.t:.2f} d", flush=True)

    x = np.r_[0.0, m.xc, LENGTH]
    initial = m.P - p0
    initial = np.r_[initial[0], initial, 0.0]
    amplitude, width = fit_step_profile(x, initial)
    rows = [f"Case{CASE}, N_f={FRACTURES}, q={rate} m3/d, L={LENGTH:g} m, cells={cells}",
            f"dt_inj={dt_inj:g} d, dt_soak={dt_soak:g} d, V=3000 m3, f_CO2=0.5",
            f"p0={p0/1e6:.2f} MPa, threshold={threshold/1e6:.3f} MPa",
            f"injection end: t={m.t:.2f} d, p_inj_peak_est={peak/1e6:.3f} MPa, "
            f"A={amplitude/1e6:.5f} MPa, a={width:.5f} m, D={D:.7e} m2/s",
            "At the closed face, the first-cell pressure approximates the boundary value.",
            "Finite-domain diffusion uses the same initial profile, zero flux at x=0, and dp(L)=0."]
    m.v_inj = 0.0
    t0 = m.t
    radii, snaps, relative = [], {}, []
    for ts in targets:
        remaining = t0 + ts - m.t
        while remaining > 1e-12:
            advance(m, min(dt_soak, remaining) * 86400.0, tol=1e-9, max_halvings=4)
            remaining = t0 + ts - m.t
        dp = m.P - p0
        dp = np.r_[dp[0], dp, 0.0]
        finite = finite_domain_solution(x, ts, D, x, initial)
        semi = redistribution(x, ts, D, amplitude, width)
        rn = threshold_radius(x, dp, threshold)
        rf = threshold_radius(x, finite, threshold)
        ra = radius(ts, D, amplitude, width, p0)
        if rn > 0:
            relative.append(abs(ra - rn) / rn)
        radii.append((ts, rn, ra, rf))
        if ts in (1, 5, 10):
            snaps[ts] = (dp, semi, finite)
        rows.append(f"ts={ts:>2} d: R_num={rn:6.2f}, R_semi={ra:6.2f}, "
                    f"R_finite={rf:6.2f}, dp_first_cell={dp[1]/1e6:.3f} MPa")
    rows.append(f"Max sampled |R_semi-R_num|/R_num for R_num > 0: "
                f"{max(relative, default=float('nan')):.2%}")
    return rows, np.array(radii), snaps, x, initial


def save_comparison(results, suffix):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for rate, (ax_profile, ax_radius), (rows, values, snaps, x, initial) in zip(
        RATES, axes, results
    ):
        ax_profile.plot(x, initial / 1e6, color="black", label="Injection end")
        for ts, (dp, semi, finite) in snaps.items():
            line, = ax_profile.plot(x, dp / 1e6, label=f"{ts} d: three-phase")
            ax_profile.plot(x, semi / 1e6, "--", color=line.get_color())
            ax_profile.plot(x, finite / 1e6, ":", color=line.get_color())
        ax_profile.axhline(0.05 * config.case_params(CASE)["p_0"] / 1e6,
                           color="gray", linewidth=0.8)
        ax_profile.set(title=f"{rate} m³/d. Dashed: half-line, dotted: finite",
                       xlabel="x (m)", ylabel="Pressure disturbance (MPa)")
        for i, label in enumerate(("Three-phase", "Half-line step", "Finite diffusion"), 1):
            ax_radius.plot(values[:, 0], values[:, i], "o-", ms=3, label=label)
        ax_radius.set(xlabel="Soak time (d)", ylabel="Threshold radius (m)", ylim=(0, None))
        for ax in (ax_profile, ax_radius):
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)

        report_name = "T08_p2_redistribution_check" if rate == 300 else "T08_q800_check"
        report = "\n".join(rows) + "\n"
        (OUT / f"{report_name}{suffix}.txt").write_text(report, encoding="utf-8")
        print(report, end="")
    fig.tight_layout()
    fig.savefig(OUT / f"T08_p2_redistribution{suffix}.png", dpi=200)
    plt.close(fig)


def save_master_curve():
    tau = np.geomspace(1e-3, 100, 360)
    samples = np.array([0.1, 1, 10, 30])
    rows = ["Half-line dimensionless master curve: tau=D*t/a^2, r=R/a",
            "A/(0.05 p0), r at tau=0.1/1/10/30"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for ratio in (2.08, 4.17, 8.33, 16.67):
        # In normalized coordinates a=1 and D=1. Convert seconds to days.
        values = [radius(t / 86400, 1.0, ratio, 1.0, 1.0, frac=1.0) for t in tau]
        sample_values = [radius(t / 86400, 1.0, ratio, 1.0, 1.0, frac=1.0) for t in samples]
        rows.append(f"{ratio:.2f}, " + "/".join(f"{value:.3f}" for value in sample_values))
        ax.semilogx(tau, values, label=f"A/(0.05 p₀) = {ratio:g}")
    ax.set(xlabel="Dimensionless time, Dt/a²", ylabel="Dimensionless radius, R/a", ylim=(0, None))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "T08_p2_master_curve.png", dpi=200)
    plt.close(fig)
    (OUT / "T08_p2_master_curve.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refined", action="store_true", help="200 cells and smaller steps through 10 d")
    args = parser.parse_args()
    cells = 200 if args.refined else 100
    try:
        results = [calculate(rate, cells, args.refined) for rate in RATES]
    except (ConvergenceError, RuntimeError) as error:
        raise SystemExit(f"T08 failed. No new reports saved. {error}") from error
    save_comparison(results, "_refined" if args.refined else "")
    if not args.refined:
        save_master_curve()


if __name__ == "__main__":
    main()
