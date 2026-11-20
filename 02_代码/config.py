# -*- coding: utf-8 -*-
"""
A 题统一参数与单位口径（T02）

三条口径（全组共用，不要各自换算）：
1. 参考状态：25 MPa、300 K。所有注入量、产量、物料按此状态计量。
2. C_n 输入为 wt%（0.5/1.0/1.5）。进入质量守恒时先除以 100 转为质量分数；
   进入经验关系 a_sigma、a_theta 时直接用 wt% 数值（系数的单位就是 per wt%）。
3. 吸附：C_a = K_d * rho_w * C_n_frac，C_a 单位 kg/kg 干岩，K_d 单位 m3/kg。

内部一律 SI：m, s, Pa, kg/m3, m2。
"""

from pathlib import Path
import openpyxl

BASE = Path(__file__).resolve().parents[1]
XLSX = BASE / "01_数据" / "A题参数表_整理.xlsx"

# ---------------------------------------------------------------- 单位换算
_MD = 9.869233e-16          # 1 mD -> m2
_MPA = 1e6                  # 1 MPa -> Pa
_Mpas = 1e-3                # 1 mPa.s -> Pa.s

# 单位字符串 -> 乘到数值上的因子；None 表示需要特殊处理
UNIT_FACTOR = {
    "m": 1.0,
    "kg/m3": 1.0,
    "K": 1.0,
    "GPa": 1e9,
    "MPa": _MPA,
    "MPa/m": _MPA,
    "mD": _MD,
    "mPa·s": _Mpas,
    "10-4MPa-1": 1e-10,     # 1e-4 / 1e6
    "MPa-1": 1e-6,
    "m2/s": 1.0,
    "nm": 1e-9,
    "mm": 1e-3,
    "wt%": 1.0,             # 保留 wt% 数值；质量分数在 case_params 里显式除 100
    "(wt%)-1": 1.0,         # 与 wt% 数值配合使用
    "(°)/(wt%)": 1.0,       # 与 wt% 数值配合使用
    "m3/kg": 1.0,
    "m/m2": 1.0,
    "m3/d": 1.0,
    "m3": 1.0,
    "d": 1.0,
    "—": 1.0,
}


def parse_num(v):
    """解析 Excel 单元格：处理 '1×10-9'、'—'、整数、浮点。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(" ", "").replace("\n", "")
    if s in ("", "—", "-"):
        return None
    s = s.replace("×10", "e").replace("×", "*")
    try:
        return float(s)
    except ValueError:
        return None


def to_si(value, unit):
    """按单位转 SI；温度单独处理为 K。"""
    if value is None:
        return None
    if unit == "°C":
        return value + 273.15
    f = UNIT_FACTOR.get(unit)
    if f is None:
        # 未知单位不静默换算，直接返回原值并让调用方注意
        return value
    return value * f


# ---------------------------------------------------------------- 读参数表
def load_params():
    """
    返回 P: {key: {"unit":..., "lower":..., "upper":..., "Case0":..., ...}}
    key 优先用符号；符号为 '—' 的行用中文参数名做 key。
    """
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    P = {}
    for sheet in ("A.1 地质与岩石力学", "A.2 流体相态与纳米颗粒", "A.3 裂缝与工程"):
        ws = wb[sheet]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or row[1] is None:
                continue
            name, sym, unit = row[1], row[2], row[3]
            key = sym if sym and sym != "—" else name
            P[key] = {
                "name": name,
                "unit": unit,
                "lower": parse_num(row[4]),
                "upper": parse_num(row[5]),
                "Case0": parse_num(row[6]),
                "Case1": parse_num(row[7]),
                "Case2": parse_num(row[8]),
                "Case3": parse_num(row[9]),
            }
    wb.close()
    return P


def load_econ():
    """经济参数，保留题面单位；另给 1e4 CNY 口径的常用量。"""
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["B.1 经济参数"]
    E = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        E[row[1]] = {"name": row[0], "value": parse_num(row[2]),
                     "unit": row[3], "note": row[4]}
    wb.close()
    return E


P = load_params()
E = load_econ()


# ---------------------------------------------------------------- 情景参数
def case_params(case=0, si=True):
    """
    取某个情景（0-3）的全部参数，转 SI。
    返回 dict：key 与 P 相同，值为换算后的数；无量纲量原样返回。
    """
    out = {}
    col = f"Case{case}"
    for k, rec in P.items():
        v = rec[col]
        out[k] = to_si(v, rec["unit"]) if si else v
    # 口径 2：C_n 输入是 wt%，这里保留 wt% 原值；需要质量分数时显式除 100
    out["C_n_wt"] = out.get("C_n")
    out["C_n_frac"] = (out["C_n_wt"] / 100.0) if out["C_n_wt"] is not None else None
    # 口径 1：温度已由 to_si 转 K
    out["T_ref"] = 300.0
    out["p_ref"] = 25e6
    return out


def adsorption_capacity(K_d, rho_w, C_n_wt):
    """口径 3：C_a = K_d * rho_w * C_n_frac，返回 kg/kg 干岩。"""
    return K_d * rho_w * (C_n_wt / 100.0)


def nano_available_ratio(phi, rho_r, rho_w, C_n_wt):
    """孔隙水中可提供的纳米质量 / 干岩质量，kg/kg。

    线性吸附若超过这个上限，说明等温线在该浓度下物理上不可实现，
    必须截断或改用 Langmuir 型。
    """
    return (C_n_wt / 100.0) * phi * rho_w / ((1.0 - phi) * rho_r)


# ---------------------------------------------------------------- 常用派生量
def hydraulic_diffusivity(case=0):
    """D = k / (phi * mu * c_t)，单相校验用。"""
    p = case_params(case)
    k = p["k_m0"]
    phi = p["φ_0"]
    mu = p["μ_o"]
    # 总压缩系数取油、水、岩石的加权；岩石项用 c_phi 代表
    c_t = p["c_o"] + p["c_w"] + p["c_φ"]
    return k / (phi * mu * c_t)


SCENARIOS = {
    1: "低渗高应力",
    2: "中渗低应力",
    3: "高渗复杂天然裂缝",
}


if __name__ == "__main__":
    # 自检：打印关键量与 Excel 对照
    lines = []
    for c in range(4):
        p = case_params(c)
        lines.append(
            f"Case{c}: km0={p['k_m0']:.3e} m2, phi={p['φ_0']}, "
            f"p0={p['p_0']:.3e} Pa, T={p['T']:.2f} K, "
            f"sigma_h={p['σ_h']:.3e} Pa, C_n_wt={p['C_n_wt']}"
        )
    lines.append(f"D(Case0) = {hydraulic_diffusivity(0):.3e} m2/s")
    ca = adsorption_capacity(P['K_d']['Case0'], P['ρ_w']['Case0'], 1.0)
    cap = nano_available_ratio(P['φ_0']['Case0'], P['ρ_r']['Case0'],
                               P['ρ_w']['Case0'], 1.0)
    lines.append(f"adsorption C_a (Case0, Cn=1wt%) = {ca:.3e} kg/kg")
    lines.append(f"available nano / rock mass     = {cap:.3e} kg/kg")
    lines.append(f"ADSORPTION OVER-CAPACITY x{ca/cap:.1f} -> 线性等温线不可直接使用，需截断或 Langmuir")
    (BASE / "03_结果与图表" / "T02_config_check.txt").write_text("\n".join(lines), encoding="utf-8")
