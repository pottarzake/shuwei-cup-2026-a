# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
p = "01_数据/附件C_Synthetic_Production_Data.xlsx"
df = pd.read_excel(p)
lines = []
lines.append(f"shape={df.shape}")
lines.append("cases=" + str(df.Case_ID.unique().tolist()))
lines.append("Cn=" + str(sorted(df["Nano_Concentration_wt%"].unique().tolist())))
lines.append("ts=" + str(sorted(df.Soak_Time_Days.unique().tolist())))
lines.append("days=" + str(df.Production_Day.min()) + "-" + str(df.Production_Day.max()))

# 360 天累计油
last = df[df.Production_Day == 360][["Case_ID","Nano_Concentration_wt%","Soak_Time_Days","Cum_Oil_m3"]]
piv = last.pivot_table(index="Soak_Time_Days", columns=["Case_ID","Nano_Concentration_wt%"], values="Cum_Oil_m3")
lines.append("\n=== 360d 累计产油 (m3) ===")
lines.append(piv.round(1).to_string())

lines.append("\n=== 各 case 最优 ts（按 360d 累计油）===")
for c, g in last.groupby("Case_ID"):
    b = g.loc[g.Cum_Oil_m3.idxmax()]
    w = g.loc[g.Cum_Oil_m3.idxmin()]
    lines.append(f"{c}: max ts={int(b.Soak_Time_Days)} Cn={b['Nano_Concentration_wt%']} Q={b.Cum_Oil_m3:.1f}; "
                 f"min ts={int(w.Soak_Time_Days)} Cn={w['Nano_Concentration_wt%']} Q={w.Cum_Oil_m3:.1f}")

lines.append("\n=== Cn 的影响 (各 case/ts 上 Q360 的极差) ===")
rng = last.groupby(["Case_ID","Soak_Time_Days"]).Cum_Oil_m3.agg(lambda s: s.max()-s.min())
lines.append(rng.groupby("Case_ID").agg(["mean","max"]).round(2).to_string())

# 首日产量与递减
first = df[df.Production_Day == 1][["Case_ID","Nano_Concentration_wt%","Soak_Time_Days","Oil_Rate_m3_d"]]
lines.append("\n=== 首日产量范围 ===")
lines.append(first.groupby("Case_ID").Oil_Rate_m3_d.agg(["min","max","mean"]).round(2).to_string())

# 数据完整性
lines.append("\n=== 缺失/重复 ===")
lines.append(f"null={int(df.isna().sum().sum())}, dup={int(df.duplicated(['Case_ID','Nano_Concentration_wt%','Soak_Time_Days','Production_Day']).sum())}")
open("03_结果与图表/数据初探.txt","w",encoding="utf-8").write("\n".join(lines))
print("done")
