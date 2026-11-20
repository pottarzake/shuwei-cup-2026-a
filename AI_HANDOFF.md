# AI_HANDOFF — A 题项目交接说明（给协作 AI）

> 读这份文件就能接手。所有路径相对于仓库根目录。
> 仓库：`github.com/pottarzake/shuwei-cup-2026-a`（私有）
> 赛题：2026 第十二届数维杯秋季赛 A 题（深部致密油 CO₂ + 纳米流体注入）
> 截止：**2026-11-24 10:00（北京时间）**
> 生成时间：2026-11-20 晚

---

## 0. 五条硬约束（违反会直接失分或违规）

1. **论文全英文**。正文、图表、附录都不能出现中文（LaTeX 注释里的中文可以）。
2. **正文 ≤25 页**，不含摘要页、目录页、附录页。当前正文 23 页。
3. **经济评价严格按附录 B**：25 MPa / 300 K 参考状态、逐日折现、B1 基准。自检值必须是 **848.25 万元**。
4. **`01_数据/` 下的 xlsx 只读**，不要覆盖；参数以 `01_数据/A题参数表_整理.xlsx` 为准。
5. **不要 force-push main**。该仓库已被多次强推，强推会让其他人的提交丢失、需要重新对基。

---

## 1. 协作规则

- **先信任对方结论，不擅自 revert**。要改，先说明证据。
- **每条关键结论标注证据等级**：
  - `verified`：有脚本 + 数值（收敛阶、质量守恒、误差）
  - `partial`：趋势或量级对，未完整校验
  - `assumed`：假设、文献外推、手算量级
- 接手代码先跑对应验证脚本，再在其上开发。
- 提交信息格式：`P1: ...`、`P3: ...`、`fix: ...`、`docs: ...`。

---

## 2. 项目结构与文件所有权

```
00_题目与规范/    原题、格式规范、提交流程、A题要点整理.md
01_数据/          附件C 数据、A题参数表_整理.xlsx（只读）
02_代码/          config.py、model/、各验证与优化脚本
03_结果与图表/    所有结果 txt/csv/png；论文插图从这里复制
04_论文/latex/    main.tex、sections/、figures/
05_文献/          28 篇 PDF + 文献映射.md
方案设计总纲.md   权威技术口径（当前 v0.23，Q1–Q23）
任务分工.md       42 张任务卡、门禁、应急预案
```

**所有权约定**：

| 路径 | 谁能改 |
|---|---|
| `02_代码/model/` | 模型负责人 |
| `02_代码/data_model.py`、`economics.py` | 数据/经济负责人 |
| `04_论文/latex/main.tex` | 统稿人（当前是队友侧） |
| `04_论文/latex/sections/` | 一人一节，改前说一声 |
| `01_数据/` | 只读 |

---

## 3. 当前状态（2026-11-20 晚）

| 模块 | 状态 | 证据等级 | 关键结果 |
|---|---|---|---|
| P1 三相正演 | 完成 | verified | 单相 vs erfc 误差 $1.2\times10^{-3}$；空间 2 阶、时间 1 阶；两相/三相质量守恒 $10^{-10}$ |
| P1 纳米输运 | 完成 | verified | 质量守恒 $10^{-10}$–$10^{-16}$；Langmuir 吸附 |
| P1 力学/损伤 | 完成 | verified（局部） | 压缩应变 2.3–3.7e-4，题给 $\varepsilon_d$ 下 **d=0**；反馈接回三相未做 |
| P2 压力再分布 | 完成 | partial | 镜像热核；5% 半径 3–4 d（q=300）/ 6–8 d（q=800）归零；解析会提前归零 |
| P3 多目标优化 | 完成 | partial | Case1 无解；Case2 NPV −164 至 −87；Case3 +365 至 +671 |
| P3 等待期 | 完成 | assumed | τ=15 d 导流降至 0.22–0.26；$c_c$ 未标定 |
| P4 数据/经济/Sobol | 完成 | verified（数据侧） | 经济 $t_s^*$=10/20/30–40 d；Case3 最高 +284；Sobol $ST_{t_s}=0.63$、$ST_{\phi_0}=0.31$ |
| 论文 | 基本完成 | — | 摘要 1 页、正文 23 页、附录 2 页；图全英文 |

---

## 4. 关键数字（唯一口径，引用时不要另算）

### 4.1 注入能力（N_f=10，T10）

| 情景 | $q_{max}$ (m³/d) | q=300 时结果 |
|---|---|---|
| Case1 | 14.6 | 0.58 d 超压 |
| Case2 | 137 | 6.6 d 超压 |
| Case3 | 1510 | 10 d 完成，52.84 MPa |

Case1 在题设区间 [100,800] m³/d 内**无可行解**；补查 q=100、V=1000，f=0.2/0.5/0.8 分别在 4.60/5.55/6.80 d 超压（计划 10 d）。

### 4.2 产量与 NPV

| 口径 | Case1 | Case2 | Case3 |
|---|---|---|---|
| P3 优化操作 NPV（万元） | 不可行 | −164 至 −87 | +365 至 +671 |
| P4 基准操作 $t_s^*$ | 10 d | 20 d | 30–40 d |
| P4 基准操作 NPV（万元） | −573 至 −866 | −315 至 −579 | +84 至 +284 |
| 盈亏平衡油价（USD/bbl） | 671–894 | 157 | 64 |

### 4.3 T31 的 $V/f$ 指数

七组三相注入+焖井，拟合 $\ln(Q/Q_0)=a\ln(V/3000)+b\ln(f/0.5)$：

$$a=-0.00721,\quad b=+0.00062,\quad R^2=0.887$$

**P3 主跑取 0**。含义：附件 C 只覆盖 $C_n$ 与 $t_s$，识别不出 $V/f$ 的增产作用；优化器选 $V=1000$、$f=0.8$ 是成本最小化结果，**不是工程建议**。

正指数对照 $(0.2,0.1)$ 下，Case3 NPV 最高点的 $V$ 从 1000 移到约 1668 m³。

### 4.4 论文页数

- 摘要 1 页、目录 1 页、正文 23 页、附录 2 页
- 页眉用 `\label{LastMainPage}` 只统计正文；附录切 `\pagestyle{plain}`

---

## 5. 当前未提交的改动（**不要 revert**）

工作区里有队友侧正在做的 B1 基准相关改动：

```
M 02_代码/07_npv_baseline.py
M 03_结果与图表/T16_npv_selfcheck.txt
M 03_结果与图表/T17_breakeven_report.txt
M 04_论文/latex/main.tex
M 04_论文/latex/sections/01_introduction.tex
M 04_论文/latex/sections/03_p1_model.tex
M 04_论文/latex/sections/05_p3_optimization.tex
M 04_论文/latex/sections/06_p4_data_econ.tex
M 04_论文/latex/sections/07_evaluation_sensitivity.tex
M 04_论文/latex/sections/08_conclusion.tex
?? 02_代码/13_b1_comparison.py
?? 03_结果与图表/T17_b1_comparison.csv
?? 04_论文/ASWBMCM2603901.pdf
```

处理方式：**先看 diff，再在其上继续**；不要用 `git checkout --` 或 `git reset --hard` 覆盖。

---

## 6. 待办（按优先级）

1. **B1 基准**：算 `NPV_B1` 与 `ΔNPV = NPV − NPV_B1`，填进 P4、摘要、结论。当前 P4 的 NPV 是绝对值，论文已注明 B1 未校核。
2. **页码口径**：队友去掉了 `\setcounter{page}{1}`，正文现在从第 3 页开始编号（摘要/目录计入总数）。如果要严格“正文从 Page 1 开始”，在 `\tableofcontents` 后加回该命令，并确认 `LastMainPage` 仍指向正文末页。
3. **提交前检查**：
   - 队号已改为 `ASWBMCM2603901`（main.tex 的 `\baominghao`）
   - 论文命名 `A+队号.pdf`，当前有 `04_论文/ASWBMCM2603901.pdf`
   - PDF ≤20 MB
   - 摘要页无学校/姓名/导师
4. **未验证项**（论文里必须标为假设）：
   - 产水 $WOR=0.5$、B1
   - 三相扰动下的鲁棒性（目前是降阶扰动 + 名义点三相复核）
   - Case1 完整可行域
   - 纳米半径的网格收敛（n=200 仍非完全收敛）
   - 等待期 $c_c$、$\tau_c$、$w_\infty$

---

## 7. 复现命令

```bash
# 核心验证
python 02_代码/02_validate_single_phase.py
python 02_代码/03_validate_two_phase.py
python 02_代码/04_validate_three_phase.py
python 02_代码/06_validate_p2.py
python 02_代码/08_validate_production.py
python 02_代码/10_p3_optimize.py
python 02_代码/12_waiting.py

# P4 脚本依赖同目录的 data_model，不能用 python -I
python 02_代码/08_model_comparison.py
python 02_代码/10_uncertainty_interpretability.py
python 02_代码/07_npv_baseline.py
python 02_代码/12_mech_data_comparison.py
python 02_代码/13_sobol_analysis.py

# 论文（两遍出目录）
cd 04_论文/latex && pdflatex main.tex && pdflatex main.tex
```

---

## 8. Git 与身份（Contributors 问题）

`ZuiXianXue` 已是协作者（write），但 **Contributors 只统计默认分支上、author email 绑定到 GitHub 账号的提交**。目前 main 的提交全记在 `pottarzake` 名下，所以看不到他。

让 `ZuiXianXue` 自己提交一次即可：

```bash
git clone https://github.com/pottarzake/shuwei-cup-2026-a.git
cd shuwei-cup-2026-a
git config user.name "ZuiXianXue"
git config user.email "他的GitHub邮箱或 ID+ZuiXianXue@users.noreply.github.com"
# 改一点东西
git add -A && git commit -m "P4: ..." && git push origin main
```

其他 Git 注意事项：

- 大文件推送失败时（`schannel: failed to receive handshake` / `early EOF`）：
  ```bash
  git config http.postBuffer 524288000
  git config http.version HTTP/1.1
  ```
- 本地是浅克隆；网络好时 `git fetch --unshallow`。
- 远端 main 曾被多次强推；拉取后先 `git log --oneline -3` 核对，不要假设本地是最新。

---

## 9. 权威资料优先级

冲突时按此顺序取数：

1. `03_结果与图表/` 下的结果文件（脚本产出，最新）
2. `方案设计总纲.md`（v0.23，Q1–Q23 的技术口径）
3. `任务分工.md`（任务状态与门禁）
4. `05_文献/文献映射.md`（文献依据）
5. 论文 `sections/*.tex`（成稿，可能滞后于前四者）

---

## 10. 一句话总结

P1–P4 的模型、结果和论文都已就位；Case3 是唯一稳健可行的情景，Case2 可行但不经济，Case1 受注入能力限制。剩下的是 B1 基准、页码口径和提交前检查，所有未验证项都已在论文中显式标注。
