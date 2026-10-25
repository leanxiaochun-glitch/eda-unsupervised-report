# EDA + Unsupervised Report Skill

一个可复用的 WorkBuddy Skill：把 **CSV / TSV / XLSX** 结构化数据转化为中文 **“可视化探索 + 无监督学习”融合报告**，输出 Markdown、图表和 PDF。

它不是某一份财务报告的硬编码模板，而是一套可迁移的方法：数据质量体检 → 严格预处理 → 单变量与交互可视化 → PCA / t-SNE → KMeans / Ward 聚类 → 标签事后验证 → 风险池与局限分析。金融、制造、医疗、零售、运营等数据均可使用，但领域结论必须根据新数据重新生成。

## 核心能力

- **标签全程隔离**：标签不参与特征处理、降维、聚类或 k 的选择，只用于事后验证。
- **数据质量体检**：量纲、整块缺失、高缺失列、常数列、ID/日期、异常字符串、±inf、极端值和不平衡。
- **固定预处理顺序**：剔除问题列 → 异常转缺失 → 中位数填补 → 1%/99% 缩尾 → RobustScaler。
- **完整可视化**：分布图、Spearman 热力图、交互浓度图、PCA、t-SNE、k 选择、簇 z 值画像、簇规模图。
- **无监督结构**：KMeans 聚类、Ward 稳健性对照、inertia/轮廓系数与业务可解释性的取舍。
- **可选事后验证**：AUC、卡方、Cramér’s V、风险池覆盖率与浓度倍数；无标签时自动跳过。
- **正式中文报告**：固定 0–10 章结构，专业作业/研究口径，不输出教学或老师视角文案。
- **可复现**：保存 `metrics.json`、分析摘要、图表和随机种子；支持一键 PDF 排版。

## 目录结构

```text
eda-unsupervised-report/
├── SKILL.md                     # 触发条件与完整工作流
├── README.md                    # 简介、安装和使用方法
├── requirements.txt             # Python 依赖
├── scripts/
│   ├── analyze_dataset.py       # 通用分析与图表生成
│   └── export_pdf.py            # Markdown + 图片导出 PDF
├── references/
│   └── report-guide.md          # 判读规则、章节要求和失败处理
└── assets/
    └── report_template.md       # 领域无关的报告模板
```

## 安装到 WorkBuddy

### 方法一：复制目录

将整个 `eda-unsupervised-report` 文件夹复制到用户级 Skill 目录：

```bash
# macOS / Linux
mkdir -p ~/.workbuddy/skills
cp -R eda-unsupervised-report ~/.workbuddy/skills/
```

Windows 可复制到 `%USERPROFILE%\.workbuddy\skills\eda-unsupervised-report`。

### 方法二：导入压缩包

如果获得 `eda-unsupervised-report.zip`，解压后按上述路径放置，或在支持导入 Skill 的客户端中选择该目录。

安装后新开会话或让客户端重新加载 Skill。无需把数据文件上传到本仓库。

## 最简单的调用方式

安装后，在 WorkBuddy 中直接说：

> 调用 `eda-unsupervised-report`，分析 `/absolute/path/data.csv`，生成可视化探索与无监督学习融合报告，并输出 PDF。

有标签时：

> 调用 `eda-unsupervised-report`，分析 `/absolute/path/data.xlsx`。标签列是 `target`，1 表示阳性；`customer_id` 是 ID。按正式课程作业口径输出 Markdown 和 PDF。

无标签时：

> 调用 `eda-unsupervised-report`，对 `/absolute/path/operations.csv` 做无监督结构探索，不使用标签，生成聚类画像和稳定性分析。

更换行业或领域时无需修改 Skill，只需提供新数据、字段含义（如有）和标签定义（如有）。

## 直接运行脚本（可选）

建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

运行分析：

```bash
python scripts/analyze_dataset.py \
  --input "/absolute/path/data.csv" \
  --output "/absolute/path/report_output" \
  --dataset-name "示例数据" \
  --label "target" \
  --positive-label 1 \
  --id-columns "customer_id,report_date" \
  --k 4
```

无标签时省略 `--label` 和 `--positive-label`。

主要参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--sheet` | `0` | XLSX 工作表名称或序号 |
| `--max-missing` | `0.80` | 剔除缺失率高于此值的列 |
| `--min-row-completeness` | `0.50` | 行在候选特征上的最低完整度 |
| `--max-features` | `120` | 最大入模特征数（按无标签方差筛选） |
| `--k` | `4` | 最终聚类数，仍需结合 k 选择结果复核 |
| `--max-rows` | `3000` | 仅用于 t-SNE 的最大行数 |
| `--seed` | `42` | 随机种子 |

脚本只输出聚合指标、摘要和图表，不导出行级数据。

生成 PDF：

```bash
python scripts/export_pdf.py \
  --input "/absolute/path/报告.md" \
  --output "/absolute/path/报告.pdf"
```

PDF 导出依赖本机 Chrome / Chromium / Edge；也可用 `--chrome` 指定浏览器路径。Markdown 中图片使用相对路径时，应保证相对于 Markdown 文件位置可访问。

## 输出文件

```text
report_output/
├── metrics.json           # 所有数值结果，报告写作的唯一数值来源
├── analysis_summary.md    # 结果核验摘要，不替代最终报告
└── figures/
    ├── fig1_distributions.png
    ├── fig2_correlation.png
    ├── fig3_interactions.png   # 有标签时生成
    ├── fig4_pca.png
    ├── fig5_tsne.png
    ├── fig6_k_selection.png
    ├── fig7_cluster_z.png
    └── fig8_clusters.png
```

最终报告由 WorkBuddy 根据这些结果和 `assets/report_template.md` 领域化撰写，避免通用脚本生成生硬或虚构的业务结论。

## 使用边界

- 默认仅使用能可靠转换为数值的列；类别变量需要另行制定合理编码方案，不能任意赋值为 1/2/3 后计算欧氏距离。
- 通用交互分析默认按中位数切分，不等同于行业标准阈值。
- t-SNE 仅用于展示，聚类在预处理后的原特征空间中完成。
- 阳性样本稀少时，卡方和浓度倍数均需谨慎解释。
- 输出是探索性结构发现与分层结果，不是监督预测模型，也不构成因果结论。
