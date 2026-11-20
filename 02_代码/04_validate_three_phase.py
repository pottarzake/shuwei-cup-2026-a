# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
"""T10：三相（水/油/气）验证——注入能力、质量守恒、相分布。

默认复核 0.5 天短测。--full 按 3000 m³ 参考体积检查完整注入期。
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "02_代码"))

from model.three_phase import ThreePhase
from model.timestepping import ConvergenceError, advance

OUT = BASE / "03_结果与图表"
OUT.mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--full", action="store_true")
parser.add_argument("--rate", type=float, default=300.0, help="reference m3/d")
parser.add_argument("--cells", type=int, default=100)
parser.add_argument("--dt", type=float, default=0.005, help="maximum step in days")
args = parser.parse_args()
if args.rate <= 0 or args.dt <= 0:
    parser.error("--rate and --dt must be positive")

duration = 3000.0 / args.rate if args.full else 0.5
lines = [
    f"N_f = 10",
    f"q_ref={args.rate:g} m3/d, f_CO2=0.5, L=40 m, cells={args.cells}, "
    f"dt_max={args.dt:g} d, t_end={duration:g} d",
    "Pressure check uses a fracture-face estimate with frozen first-cell mobility. "
    "The screening model treats CO2 as insoluble.",
]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))


def screen_case(case):
    model = ThreePhase(L=40.0, n=args.cells, stretch=0.5, case=case, fg_inj=0.5,
                       q_inj_ref=args.rate / 86400.0, N_f=10)
    pressure_cap = model.prm["p_max"]
    peak = model.prm["p0"]

    def record_pressure():
        nonlocal peak
        peak = max(peak, model.injection_pressure())
        return peak <= pressure_cap

    status = "completed without estimated overpressure"
    while duration - model.t > 1e-12:
        step_days = min(args.dt, duration - model.t)
        try:
            result = advance(model, step_days * 86400, tol=1e-9, on_step=record_pressure)
        except ConvergenceError as error:
            status = f"numerical failure: {error}"
            break
        if result is None:
            status = "estimated pressure limit exceeded"
            break

    mb = model.mass_balance()
    scale = np.maximum.reduce([np.abs(mb["initial"]), np.abs(mb["in"]), np.abs(mb["out"]),
                              np.abs(mb["stored"] - mb["initial"])])
    scale = np.maximum(scale, np.finfo(float).eps * np.maximum(mb["initial"], 1.0))
    relative = np.abs(mb["residual"]) / scale
    return model, peak, status, relative


for case in (1, 2, 3):
    m, peak, status, relative = screen_case(case)
    lines.append(
        f"Case{case}: t_stop={m.t:.3f} d, {status}, "
        f"p_face_peak={peak/1e6:.2f} MPa (limit {m.prm['p_max']/1e6:.1f}), "
        f"p_first_cell={m.P[0]/1e6:.2f} MPa, Sg_max={m.Sg.max():.3f}, "
        f"mass_rel(w/o/CO2)={relative[0]:.2e}/{relative[1]:.2e}/{relative[2]:.2e}"
    )
    label = f"Case{case}, {m.t:.3g} d"
    axes[0].plot(m.xc, (m.P - m.prm["p0"]) / 1e6, label=label)
    axes[1].plot(m.xc, m.Sg, label=label)

axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("p - p0 (MPa)")
axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("S_g")
for ax in axes:
    ax.grid(alpha=.3); ax.legend(fontsize=8)
name = (f"T10_full_q{args.rate:g}_n{args.cells}" if args.full
        else "T10_three_phase")
fig.tight_layout()
fig.savefig(OUT / f"{name}_profiles.png", dpi=200)
plt.close(fig)
report = "\n".join(lines) + "\n"
(OUT / f"{name}_check.txt").write_text(report, encoding="utf-8")
print(report, end="")
