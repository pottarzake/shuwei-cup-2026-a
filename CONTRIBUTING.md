# 协作指南

> 两人、四天、一个私有仓库。`main` 随时可运行、可编译；各自开分支；每天至少同步一次。

---

## 1. 环境准备

需要：Git、Python 3.11+、LaTeX（TeX Live 或 MiKTeX）、VS Code（推荐装 LaTeX Workshop 与 Python 插件）。

```bash
git clone https://github.com/pottarzake/shuwei-cup-2026-a.git
cd shuwei-cup-2026-a

# Windows 下让中文文件名正常显示
git config core.quotepath false
```

论文模板在 `04_论文/latex/`，**推荐用 pdfLaTeX 编译**（模板也支持 XeLaTeX）。首次编译前先确认 `swmcmthesis.cls` 和 `swmcmthesis.cfg` 与 `demo.tex` 在同一目录。

模板顶部要改两处：

```latex
\baominghao{SWBMCM2600001}   % 改成自己的队号
\zubie{ABCD}                 % 改为 A
```

---

## 2. 分支模型

只有一个长期分支：`main`。

| 分支 | 用途 | 生命周期 |
|---|---|---|
| `main` | 稳定版本，随时可编译 | 长期 |
| `feat/p1-equations` | 新功能 | 合并后删除 |
| `fix/data-split` | 修 bug | 合并后删除 |
| `docs/readme` | 文档 | 合并后删除 |

规则：

- **不要直接往 `main` 推**，除非只是改错别字这类一行改动
- 没写完也要每天 push 到远端分支，防止本地丢失
- 合并走 Pull Request，另一人看一眼再合；紧急情况下可直接合并，但要在群里说一声

---

## 3. 日常流程

**开工前：**

```bash
git switch main
git pull --rebase origin main
git switch -c feat/p3-nsga2      # 建自己的分支
```

**写代码 / 写论文：**

```bash
git add -A
git commit -m "P3: NSGA-II 初版跑通，Pareto 前沿保存在 03_结果与图表/"
git push -u origin feat/p3-nsga2
```

**收工前：**

```bash
git switch main
git pull --rebase origin main
git merge feat/p3-nsga2
git push origin main
```

或者到 GitHub 上开 PR，让队友 review 后再 squash merge。

---

## 4. 提交信息规范

格式：`<范围>: <做了什么>`

| 范围 | 含义 |
|---|---|
| `P1`–`P4` | 对应题目第 1–4 问 |
| `data` | 数据清洗、参数表 |
| `docs` | 文档、README |
| `paper` | 论文正文、图表 |
| `fix` | 修 bug |

示例：

```
P1: 补全 Biot 有效应力耦合项
P2: 无量纲群推导，补量级表
P3: 代理模型改为 Kriging，单次评估耗时降到 0.1 s
P4: 按组划分训练/测试，留出 t_s=55 d 作外推
paper: 完成 Problem 2 小节初稿
fix: 修复 NPV 折现指数少除 365 的问题
```

一次提交只做一件事。不要写「update」「修改」这种无法回溯的信息。

---

## 5. 冲突处理

推送被拒（`rejected`）时：

```bash
git pull --rebase origin main
```

出现冲突：

1. 打开冲突文件，找到 `<<<<<<<`、`=======`、`>>>>>>>` 标记
2. 保留正确内容，删掉三行标记
3. `git add <文件>`
4. `git rebase --continue`

拿不准就停下来问，**不要**用 `git push -f` 覆盖远端。

---

## 6. 文件所有权

二进制文件无法自动合并，必须约定唯一维护人。

| 路径 | 规则 |
|---|---|
| `01_数据/*.xlsx` | **只由一人修改**。另一人需要改，先说一声，改完立刻提交 |
| `01_数据/附件C_*.xlsx` | 只读，任何人不得覆盖 |
| `02_代码/` | 按模块拆文件，避免两人同时改同一个 `.py` |
| `04_论文/latex/demo.tex` | 拆成多个 `\input{}` 分节文件，**一人一节**；`demo.tex` 主文件只由一人改 |
| `03_结果与图表/` | 只追加，不覆盖别人的图；文件名带问题号，如 `p3_pareto_case1.png` |
| `README.md` / `CONTRIBUTING.md` | 都可以改，改完说一声 |

论文分节建议：

```
04_论文/latex/
├── demo.tex          # 主文件：只管 \input，不写正文
├── sections/
│   ├── 01_intro.tex
│   ├── 02_assumptions.tex
│   ├── 03_p1_model.tex
│   ├── 04_p2_scaling.tex
│   ├── 05_p3_optimization.tex
│   ├── 06_p4_data_econ.tex
│   ├── 07_sensitivity.tex
│   └── 08_conclusion.tex
```

---

## 7. 数据文件

- `.xlsx` 是二进制，Git 无法合并，冲突时只能二选一
- 单文件不要超过 50 MB；超过就放网盘并在 README 里写链接
- `01_数据/附件C_Synthetic_Production_Data.xlsx` 约 1.2 MB，正常入库
- 中间结果（大 CSV、模型权重）放 `03_结果与图表/`，超过 20 MB 的不要提交，加进 `.gitignore`

---

## 8. 时间表（倒排）

| 时间 | 目标 |
|---|---|
| 11/20 下午 | 仓库就绪；P1 方程体系定稿；确定分工 |
| 11/21 | P2 无量纲化与解析解；P3 一维正演跑通 |
| 11/22 | P3 代理模型 + NSGA-II；P4 数据模型 |
| 11/23 | 结果整合、图表定稿、论文初稿 |
| 11/24 08:00 | 论文定稿，停止改内容 |
| 11/24 09:00 | 提交完毕 |

**留 2 小时缓冲。** 截止时间是 11/24 10:00，不要卡点提交。

---

## 9. 禁止事项

- 提交 API key、账号密码、`.env` 文件
- `git push -f` 覆盖 `main`
- 把赛题 PDF、论文、代码转发到公开渠道（比赛期间仓库保持私有）
- 论文正文与附录出现学校、姓名、导师等身份信息
- 提交 `.venv/`、`__pycache__/`、LaTeX 编译中间产物（已在 `.gitignore` 中排除）

---

## 10. 常见问题

**中文文件名显示成 `\346\226\207...`**
```bash
git config core.quotepath false
```

**xlsx 冲突了**
放弃本地版本，用远端覆盖后手动重做改动：
```bash
git checkout --theirs 01_数据/xxx.xlsx
```

**LaTeX 编译报找不到 `swmcmthesis.cls`**
确认当前工作目录是 `04_论文/latex/`，且 `.cls`、`.cfg` 与主 `.tex` 同级。

**编译出来中文乱码**
换 XeLaTeX 试一次；仍不行检查文件是否以 UTF-8 保存（VS Code 右下角编码）。

**推送提示没有权限**
确认已被邀请为仓库协作者，且接受过邀请邮件。
