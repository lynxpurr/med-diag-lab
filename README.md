# med-diag-lab — 中文病历诊断微调实验场

**English summary.** med-diag-lab is a small, fully documented fine-tuning experiment on a public
Chinese medical-record diagnosis dataset (1,337 cases, 710 diseases) using Qwen3 and ms-swift on a
single 16 GB GPU. It measures what capacity, training data and preference tuning (DPO) each buy and
what they cost, under pre-registered verdicts, a disease-level leakage-free split, and a deterministic
rule-based evaluator that must pass a gold self-check before any model is scored. The main
deliverable is a "lever × failure-axis" map ([notes/map.md](notes/map.md)). Code: Apache-2.0. The
dataset and model weights are not redistributed. All documentation is in Chinese.
**Research use only — not for clinical use.**

> ## ⚠️ 非临床用途
>
> - 本仓库是机器学习方法学实验。产出物不是医疗器械，**不能用于真实患者的诊断、治疗或任何临床决策**。
> - 没有做任何临床有效性验证。全部读数只说明模型在一个 1337 例的公开数据集上如何表现。
> - 仓库不分发模型权重，也不提供面向患者的使用入口。
> - 随仓的合成“无病”体检评测集由大模型起草，未经医生审核。

## 这是什么

**项目状态**：实验已于 2026-09-30 完成并归档，本仓库是发布快照。

在一个很小的中文病历诊断数据集上，用 Qwen3 + ms-swift 回答一个问题：**容量、训练数据、偏好这三类杠杆各自能换来什么，代价是什么。**

实验分六波推进。每一波开跑之前，先把判据和“两种结果各说明什么”写进 [PLAN.md](PLAN.md)；跑完按原文判决。偏离预期时只改后续实验的设计，不改已写下的判决。本仓库是单提交快照，这些先后关系依据文档内的日期，无法从 git 历史核验。

| 波 | 完成日期 | 做了什么 |
|---|---|---|
| MED-0 | 2026-09-27 | 数据切分，三种训练目标形态 |
| MED-A | 2026-09-28 | Qwen3 五档零样本基线 |
| MED-B | 2026-09-29 | SFT 三臂（先在 1.7B 上快筛，再在 14B 上做主实验） |
| MED-C | 2026-09-29 | 合成数据增强三臂，在 SFT 冠军上续训 |
| MED-D | 2026-09-30 | DPO 剂量阶梯（1 / 3 / 9 步） |
| MED-E | 2026-09-30 | seen 与 unseen 的差距量化、长尾分层，地图定稿 |

**规模与边界**（读结论前请先看这里）：单数据集，单随机种子，单模型族（Qwen3）；推理与评测都是 4bit 量化口径；评测集只有 80 / 60 / 40 / 20 / 10 条，val-reg 上 1 分约等于 1 道题。

## 结论速览

| 配方 | val-reg 疾病 F1（seen） | test-ood 疾病 F1（unseen） | 备注 |
|---|---|---|---|
| Qwen3-14B 零样本 | 0.412 | 0.072 | 基线 |
| SFT，官方目标形态（f1） | 0.800 | 0.133 | 对照基线 |
| SFT，凝练依据（f3c） | 0.840 | 0.083 | 主判据高 4 分，但依据覆盖从 0.976 掉到 0.905 |
| 续训，加 12 条无病合成数据（cn-low） | 0.863 | 0.067 | 无病轴满分，unseen 减半 |
| SFT + DPO 3 步（dpo-s3） | 0.812 | 0.133 | 各轴均无回退 |

几条值得带走的观察（详见 [notes/map.md](notes/map.md)）：

- **seen 与 unseen 的差距是结构性的。** SFT 把 seen F1 从 0.41 提到 0.80，unseen 只从 0.07 到 0.13。在冠军上叠加续训 epoch，会把差距放大到 12.9 倍。
- **泛化体现在组织证据，不体现在叫出病名。** unseen 侧的依据覆盖在 11 个配方里有 10 个高于零样本，病名命中几乎为零。
- **同一个信号，用在不同杠杆上结果不同。** 按 score 过滤训练目标，对 SFT 是零增量；同样的 score 信号做成偏好对，经 DPO 带来 +1.2 分。
- **不对症的增强数据是负资产。** 给已经满分的鲁棒轴补数据没有收益，反而让“无病”这一轴回退。
- **DPO 的剂量是开关型的。** 1 步无效，3 步饱和，9 步不再变化。
- **数值幻觉的主要形态是“填入正常参考值”。** 模型把教科书里的正常值当成检验结果写出来，看起来毫不离谱。

## 方法上可复用的部分

- **预注册判决。** 判据先于实验写死，并预先写好每种结果的含义，避免事后合理化。
- **疾病级零泄漏切分。** test-ood 只取“所有疾病在整个语料里只出现过一次”的记录，所以训练集在构造上不可能见过这些病名（`tools/build_dataset.py` 里的断言）。
- **确定性评测器与 gold 自校准门。** 评分只靠正则与集合运算，不调用 LLM 评审。判分口径改动之后，评测器必须先给标准答案打出满分，才允许给模型打分（`tools/run_eval_med.py gold`）。
- **证据只增不改。** 评分与推理产物已存在就拒绝覆盖；报告只允许追加补记。
- **分轴读数。** 疾病命中、格式、依据覆盖、鲁棒、无病、数值幻觉各读各的，主判据与护栏分开判。

## 仓库结构

```
├── tools/          构建、评测、训练、对拍脚本
├── notes/          map.md（地图）· playbook.md（方法论）· lessons.md（教训）
├── reports/        六波实验报告
├── research/       数据集调研与阶段路线分析
├── evidence/       数值证据：评分 JSON、训练参数快照、逐步训练日志、长尾分层
├── data/           fetch_raw.sh · PROVENANCE.md · raw/SHA256SUMS · medc/eval-neg*.jsonl
├── PLAN.md · HANDOFF.md · SPEC-INDEX.md    过程账本
└── GLOSSARY.md · LICENSE · NOTICE · requirements*.txt
```

## 快速开始

### 只用标准库：下载数据、重建夹具、跑评测器自校准

```bash
git clone https://github.com/lynxpurr/med-diag-lab.git && cd med-diag-lab
bash data/fetch_raw.sh                                                  # 下载上游数据并校验 SHA256
python3 tools/build_dataset.py                                          # 切分 + 三种训练目标 + 评测夹具（内含零泄漏断言）
python3 tools/run_eval_med.py gold val-reg test-ood test-rob neg negn   # 评测器自校准，五集应全部“过”
```

这三步只用 Python 标准库，在 3.12 和 3.14 上验证过。`build_dataset.py` 是确定性的：重建出的训练与评测夹具，和实验所用版本逐字节一致。

### 完整流水线：偏好对与合成增强（需要 ms-swift 和 Qwen3 模型目录）

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt          # 或 pip install -r requirements-lock.txt 复现完整环境

export MODELS_DIR=/path/to/models        # 其下是 Qwen3-14B/、Qwen3-1.7B/ 等模型目录
                                         # swift 取 PATH 上的可执行文件；否则另设 SWIFT_BIN=/path/to/venv/bin/swift

python tools/profile_dpo_lengths.py      # 偏好对长度实测 → data/medd/length-profile.json（CPU，约 2 分钟）
python tools/build_dpo_pairs.py          # → data/medd/dpo-pairs.jsonl（300 对）
python tools/check_dpo_med.py            # 对拍：格式 / 公共前缀 / 长度 / 出口形态 / 组装保真，须全绿
python tools/augment_med_c.py            # 合成增强三臂 → data/medc/aug-*.jsonl
python tools/augment_med_c.py low        # cn-low 回退分支的素材
python tools/check_template_med.py ref && python tools/check_template_med.py swift   # 模板对拍
```

两条安装路径都在 2026-11-27 于全新的 Python 3.12.13 环境里从零安装，并跑通了不依赖 GPU 的流水线，每条约 4 分钟（当时 pip 缓存里已有大部分安装包）。

> **安全说明**：两个文件里的直接依赖钉在 2026 年 9 月实验结束时的版本，没有做安全更新。
>
> - 按 `requirements.txt` 安装时，传递依赖会取当前版本。2026-11-27 实测，装出的环境里仍有已公开公告的包是 4 个：`datasets`（中危；`ms-swift==4.5.3` 要求 `datasets<4.8.5`，升不到修复版本）、`fsspec`、`setuptools`、`torch`。本仓的代码不直接使用这几个库。
> - `requirements-lock.txt` 是实验环境的精确快照，其中 10 个包共有 94 条后来发布的公告，含严重级别。只在需要逐位复现实验环境时使用。
>
> 请在隔离环境里安装，不要用这套环境处理不可信数据。

### 训练与评测（16 GB 单卡）

```bash
python3 tools/run_eval_med.py prep val-reg test-ood test-rob neg negn     # 渲染推理输入（与模型无关）
bash tools/run_med_b.sh 14b f1            # SFT 的一臂；与常驻服务共享的 16 GB 卡上实测约 1 小时 40 分
python3 tools/run_eval_med.py infer sft-14b-f1 val-reg --adapters output/med-b/14b-f1/checkpoint-219 --out output/med-b
python3 tools/run_eval_med.py score  sft-14b-f1 val-reg --out output/med-b
python3 tools/run_eval_med.py table --out output/med-b
```

其余形态见各脚本的头注：三臂评测 `run_med_b4.sh`、续训 `run_med_c.sh` + `run_med_c3.sh`、DPO `run_med_d.sh` + `run_med_d3.sh`、零样本矩阵 `run_med_a_matrix.sh`。训练脚本在 GPU 空闲显存低于阈值时拒绝启动，产物已存在时跳过。

### 核对报告里的数字

```bash
python3 tools/run_eval_med.py table --out evidence/med-d      # 从公开的评分 JSON 重新聚合主表
EVIDENCE_ROOT=evidence python3 tools/med_e_stratify.py        # 重算长尾分层 → output/med-e/stratify.json，
                                                              # 应与 evidence/med-e/stratify.json 相同
```

`evidence/` 的文件清单和 tag 对照见 [evidence/README.md](evidence/README.md)。

## 阅读指南

1. [notes/map.md](notes/map.md)：杠杆 × 失败轴地图，项目的主交付。每个配方给出读数、代价、适用情形，以及改善归属于哪一层。
2. `reports/`：六波实验报告，按 med-0 到 med-e 的顺序读。每份报告给出该波的读数明细，以及按预注册原文写下的判决。
3. [notes/playbook.md](notes/playbook.md) 与 [notes/lessons.md](notes/lessons.md)：可操作的方法论，以及踩过的坑。

过程账本按原样保留，供追溯：[PLAN.md](PLAN.md)（目标、决策、风险、预注册判决）、[SPEC-INDEX.md](SPEC-INDEX.md)（每类规格的事实源在哪）、[HANDOFF.md](HANDOFF.md)（会话交接与阶段卡）。它们是为作者本人和 AI 助手的连续工作会话写的，密度高。读之前先看 [GLOSSARY.md](GLOSSARY.md) 的编号对照；实验的时间线见 [HANDOFF.md](HANDOFF.md) 的 §3（按波次）和 §4（逐日经过）。

## 数据、许可与出处

- **代码与文档**：Apache-2.0，见 [LICENSE](LICENSE)；第三方说明见 [NOTICE](NOTICE)。
- **上游数据集不分发。** 实验基于 ModelScope 上的 `zhangmingme/medical-record-diagnosis-chinese`（1337 例中文病历）。上传者声明为 Apache-2.0，但没有说明病例的来源，这一点无法独立核实。因此本仓库不收录数据集本体，也不收录由它派生的训练与评测夹具，只提供下载脚本、指纹和重建工具。出处与指纹见 [data/PROVENANCE.md](data/PROVENANCE.md)。
- **随仓的唯一数据**是 `data/medc/eval-neg.jsonl` 与 `data/medc/eval-negn.jsonl`：30 条合成的“无病”体检文本，用来评测模型会不会给健康人安上诊断。它们由大模型起草，数值参考区间取自中国卫生行业标准 WS/T 404-2012 与 WS/T 405-2012，**未经医生审核**。记录元数据里的 `reverse_probe` 字段停在起草时的“待跑”状态；探针实际已跑，结果见 `reports/med-b-sft.md`。过程文档里把 `eval-neg.jsonl` 的 20 条称作“人工形态”，意思是自由起草，不是人手撰写。
- **模型权重与 LoRA 适配器不发布。**
- **AI 辅助说明。** 本项目的脚本、数据构建、合成评测集与文档，由作者在 AI 编程助手的协助下完成。X4 工作表里的“人核”判定是模型初判，不是人工复核（表内已标注）。`eval-negn.jsonl` 元数据里的“人核”同样如此：是 AI 助手代作者逐条核查，不是人工或医生审核。

## 公开版说明

本仓库是作者私有工作副本的发布快照，所有读数与结论未作改动。相对私有版本：

- 工具脚本的模型目录与 swift 路径改为环境变量 `MODELS_DIR`、`SWIFT_BIN`（原来写的是本机路径）。`med_e_stratify.py` 新增 `EVIDENCE_ROOT`，用来指定评分 JSON 所在目录。
- 文档里的本机绝对路径、主机标识、同机其他项目的服务名，替换成了通用描述；对项目所有者的称呼统一改为“作者”。
- 作者另外两个未公开的仓库，在正文里用代称“上游方法学仓”“姊妹域实验仓”指代（见 [GLOSSARY.md](GLOSSARY.md)）。全文不出现任何私有仓库的名称、提交哈希或分支名。
- `evidence/` 是运行产物目录 `output/` 中数值证据的导出，路径同构：各集评分 JSON（去掉了模型回复正文）、训练参数快照、逐步训练日志、长尾分层结果。文档里提到的 `output/…` 路径，在公开版里对应 `evidence/…`。
- [research/2026-09-28-x4-pairs-worksheet.md](research/2026-09-28-x4-pairs-worksheet.md) 里逐字引用上游分析链的两列，只保留了项数。
- 未收录：上游数据集及其派生夹具（可由脚本重建）、模型回复全文与运行日志、LoRA 权重、评测器开发期的评分文件（`output/med-a/score-dev/`）、AI 助手用的项目操作手册（`.zcode/skills/`，文档中提到的“skill §N”指它），以及私有仓的提交历史。
