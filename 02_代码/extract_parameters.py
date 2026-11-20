# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
# OpenAI Codex (GPT-6 family, exact release date unconfirmed) assisted review or edits on 2026-11-20.
from pathlib import Path

import pdfplumber, json, re
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

BASE = Path(__file__).resolve().parents[1]
PDF = BASE / "00_题目与规范" / "A题_原题.pdf"
OUT = BASE / "01_数据" / "A题参数表_整理.xlsx"

param, econ = [], []
with pdfplumber.open(PDF) as pdf:
    for i in range(5, 11):
        for t in pdf.pages[i].extract_tables():
            if not t: continue
            h = t[0]
            if h and len(h) == 8 and h[0] and "Parameter" in h[0]:
                param += t[1:]
            elif h and len(h) == 5 and h[0] == "Item":
                econ += t[1:]

assert len(param) == 65, len(param)
assert len(econ) == 11, len(econ)

CN = [
("埋深","H_d"),("储层厚度","h"),("原始地层压力","p_0"),("地层温度","T"),
("基质渗透率","k_m0"),("初始孔隙度","φ_0"),("杨氏模量","E"),("泊松比","ν"),
("上覆岩层压力","σ_v"),("最大水平主应力","σ_H"),("最小水平主应力","σ_h"),
("破裂压力梯度","G_f"),("破裂压力","p_f"),("井底压力上限","p_max"),("岩石密度","ρ_r"),
("Biot 系数","α"),("应力敏感系数","γ_k"),("孔隙度压力系数","c_φ"),("损伤阈值","ε_d"),
("损伤软化参数","β_d"),
("原油密度","ρ_o"),("原油黏度","μ_o"),("地层水密度","ρ_w"),("地层水黏度","μ_w"),
("CO₂ 黏度（储层条件）","μ_CO2"),("原油压缩系数","c_o"),("地层水压缩系数","c_w"),
("初始含水饱和度","S_w0"),("束缚水饱和度","S_wr"),("残余油饱和度","S_or"),
("油相相对渗透率端点","k_ro0"),("水相相对渗透率端点","k_rw0"),("油相指数","n_o"),
("水相指数","n_w"),("基质毛管压力系数","A_c"),("CO₂ 临界温度","T_c"),
("CO₂ 临界压力","p_c"),("参考状态 CO₂ 密度","ρ_CO2,ref"),("CO₂ 在油中亨利常数","H_o"),
("CO₂ 在水中亨利常数","H_w"),("原油膨胀系数","β_sw"),("降黏指数","m_μ"),
("CO₂ 在油中扩散系数","D_CO2,o"),("CO₂ 在水中扩散系数","D_CO2,w"),("纳米颗粒直径","d_n"),
("纳米颗粒浓度（设计变量）","C_n"),("纳米颗粒水相扩散系数","D_n"),
("纳米颗粒吸附分配系数","K_d"),("界面张力降低系数","a_σ"),("接触角改变系数","a_θ"),
("水力裂缝半长","L_f"),("缝高","h_f"),("缝宽","w_f"),("裂缝渗透率","k_f"),
("裂缝孔隙度","φ_f"),("天然裂缝密度","ρ_nf"),("注入速率（设计变量）","q_inj"),
("总注入量（设计变量）","V_inj"),("CO₂ 体积分数（设计变量）","f_CO2"),
("焖井时间（设计变量）","t_s"),("注入时机（设计变量）","τ_inj"),("井底流压","p_wf"),
("生产时长","—"),("注入+焖井时长上限","—"),("单井日产液上限","—"),
]
assert len(CN) == 65

def num(v):
    if v is None: return None
    s = str(v).strip().replace("\n","")
    if s in ("", "—", "-"): return "—"
    try:
        f = float(s)
        return int(f) if f == int(f) and "." not in s else f
    except ValueError:
        return s

HEAD_FILL = PatternFill("solid", fgColor="1F4E79")
HEAD_FONT = Font(color="FFFFFF", bold=True, size=11)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

def style_sheet(ws, headers, widths):
    for c, (h, w) in enumerate(zip(headers, widths), 1):
        cell = ws.cell(row=1, column=c)
        cell.fill, cell.font = HEAD_FILL, HEAD_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.column_dimensions[get_column_letter(c)].width = w
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=len(headers)):
        for cell in row:
            cell.border = BORDER
            if cell.row > 1:
                cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"

PHEAD = ["序号","参数","符号","单位","下限","上限","Case0","Case1","Case2","Case3"]
EHEAD = ["项目","符号","数值","单位","备注"]
wb = Workbook()

# --- A.1 ---
ws = wb.active; ws.title = "A.1 地质与岩石力学"
ws.append(PHEAD)
for k, (r, cn) in enumerate(zip(param[0:20], CN[0:20]), 1):
    ws.append([k, cn[0], cn[1], r[1], num(r[2]), num(r[3]), num(r[4]), num(r[5]), num(r[6]), num(r[7])])
style_sheet(ws, PHEAD, [6,26,14,12,9,9,9,9,9,9])

# --- A.2 ---
ws = wb.create_sheet("A.2 流体相态与纳米颗粒")
ws.append(PHEAD)
for k, (r, cn) in enumerate(zip(param[20:50], CN[20:50]), 1):
    ws.append([k, cn[0], cn[1], r[1], num(r[2]), num(r[3]), num(r[4]), num(r[5]), num(r[6]), num(r[7])])
style_sheet(ws, PHEAD, [6,28,14,12,9,9,9,9,9,9])

# --- A.3 ---
ws = wb.create_sheet("A.3 裂缝与工程")
ws.append(PHEAD)
for k, (r, cn) in enumerate(zip(param[50:65], CN[50:65]), 1):
    ws.append([k, cn[0], cn[1], r[1], num(r[2]), num(r[3]), num(r[4]), num(r[5]), num(r[6]), num(r[7])])
style_sheet(ws, PHEAD, [6,30,14,12,9,9,9,9,9,9])

# --- B.1 ---
ws = wb.create_sheet("B.1 经济参数")
ws.append(EHEAD)
ECN = [("原油价格","P_oil"),("汇率","e"),("折合原油价格","P_oil*"),("纳米材料单价","c_nano"),
       ("CO₂ 井口价","c_CO2"),("注入作业日费","c_inj"),("焖井维护日费","c_soak"),
       ("压裂施工费","C_frac"),("可变生产成本","c_prod"),("采出水处理费","c_water"),("年折现率","r")]
for r, cn in zip(econ, ECN):
    ws.append([cn[0], cn[1], num(r[2]), str(r[3]).replace(chr(10), " "), str(r[4]).replace(chr(10), " ")])
style_sheet(ws, EHEAD, [22,14,12,16,60])

# --- 说明 ---
ws = wb.create_sheet("说明")
notes = [
 ["数据来源","《2026 “ShuWei Cup” A Problem》附录 A 表 A.1–A.3、附录 B 表 B.1，由 PDF 表格自动抽取后人工核对"],
 ["单位约定","孔隙度/饱和度/指数等无量纲量单位记为 “—”；10-4 MPa-1 表示 1e-4 MPa^-1"],
 ["破裂压力","p_f = G_f × H_d，表中已按此式计算并保留一位小数"],
 ["井底压力上限","p_max = min(0.95 p_f, 60 MPa)"],
 ["设计变量","C_n、q_inj、V_inj、f_CO2、t_s、τ_inj 在问题 3 中作为优化变量"],
 ["三情景","Case1 低渗高应力；Case2 中渗低应力；Case3 高渗且天然裂缝发育；Case0 为基准算例"],
 ["区间含义","下限—上限仅用于参数扰动、敏感性分析与不确定性情景，不代表概率分布"],
 ["优化方案提示","未考虑应力敏感性时可取 γ_k = 0、c_φ = 0，且缝宽与缝渗保持不变"],
]
for n in notes: ws.append(n)
style_sheet(ws, ["条目","内容"], [16,95])
ws.column_dimensions["B"].width = 95

wb.save(OUT)
print("saved:", OUT)
