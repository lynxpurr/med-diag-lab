# MED-0 数据基建 — 落证报告

> 波次：MED-0（零 GPU）· 日期：2026-09-27 · 构建器：`tools/build_dataset.py` · seed 20260925
> 预注册判决：R0a/R0b/R0c（PLAN §4 MED-0 原文）· 本文是唯一 run 证据，只增不改。

## 0. 结论先行

**三判决全过，数据夹具就绪可开工 MED-A。** 1337 → 去重 1314 → train 1154 / val-reg 80 /
test-ood 60 / test-rob 40（20 基材 × 乱码/极简双条件）；三训练形态 F1/F2/F3c 各 1154 行同 N 同
user（臂间可比性由构造保证）。

## 1. 去重与切分

| 步骤 | 实测 |
|---|---|
| 源 | 1337 例（与官方 SFT 版按下标 1:1 对齐，逐条 analyses 计数验证） |
| 去重（condition 全文哈希） | **-23 条冗余（22 组）**，与调研预测一致 → 1314 |
| test-ood | 60 例，抽自全 singleton 病池 381（按 字段组合×性别 双层配额）；性别 男29/女27/未知4 |
| val-reg | 80 例（全部疾病频次 ≥4 的池内贪心抽取，保 R0c）；男41/女31/未知8 |
| rob_base | 20 例（同池，与 val-reg 不相交）→ 乱码+极简 40 条评测项 |
| train | 1154 例（男575/女492/未知86/**幼儿1**——源数据 gender 含"幼儿"值，非构建引入） |

**切分读数**：train 覆盖 649/710 唯一疾病；不在 train 的 61 病 = test-ood 的 singleton 病集
（R0a 断言的另一面，数量对账吻合）。val-reg/test-ood 的分析条数 mean 4.8/5.4、每例病数
mean 1.00/1.02——与全库分布（5.0/1.05）一致，无切分偏斜。

## 2. 三训练形态（data/med0/swift-train-*.jsonl）

| 形态 | 构成 | 行数 | 回退 | 目标长度 |
|---|---|---|---|---|
| F1 官方式 | user/target 逐字节复用官方 SFT 行 | 1154 | — | max 521 字 |
| F2 质量过滤 | 仅 score≥2 分析入分条依据 | 1154 | **0**（train 内无全低分记录） | max 521 字 |
| F3-concise | `<diseases>` 同官方 + 依据改用 diagnoses.reason | 1154 | **13**（= diagnoses=None 全记录，回退 F1 目标，保三臂同 N） | max 389 字 |

三臂同 N、同 user、同 system prompt（从官方 SFT 版提取的全局唯一串）；差异只在 assistant 目标。

## 3. 判决

- **R0a 切分零泄漏：过**（断言级）。train 疾病集 ∩ test-ood 疾病集 = ∅；val-reg/rob_base 疾病集 ⊆ train。
- **R0b fidelity：过**（断言级）。条数守恒（1314 = 1154+80+20+60）· 角色序 · user 逐字 ·
  **F1 目标自建构造器重建与官方逐字节相等（1154/1154）**——F2/F3c 与 F1 的后续读数差不可能来自构造器。
- **R0c 评测病不抽干：过**（断言级）。val-reg ∪ rob_base 的每个疾病在 train 剩 ≥2 例。

## 4. 评测夹具（data/med0/eval-*.jsonl）

- eval-val-reg 80 / eval-test-ood 60：user（官方渲染逐字）+ gold（diseases/conclusions/reasons/analyses 全量）。
- eval-test-rob 40：乱码条件（确定性字符表替换，保行结构）预期类别=数据异常；极简条件（仅
  性别/年龄/主诉行）预期类别=推测或缺少关键诊断信息。**"无病"第三条件未建**（X2：可信合成需
  设计，MED-C 立项时定，见 PLAN §4 MED-C）。
- 评测器 `tools/run_eval_med.py` 未建（MED-A 开工件）；`tools/normalize.py` 已就位（SYNONYMS
  起点为空，评测开发期按错误分析追加制填充）。

## 5. 残留与移交

- f3c 的 13 条回退意味着 F3 臂在 diagnoses=None 记录上与 F1 同目标——臂间差异被这些记录稀释
  （13/1154 ≈ 1.1%，可忽略，如实记）。
- test-ood 的"unseen"语义 = 疾病级 unseen（病历文本仍同域同风格）；不要把它读成跨域 OOD。
- 下一步：MED-A 开工卡（HANDOFF §2）——评测器先在 gold 上自校准再上模型。

## 6. 补记（2026-09-28）：重复组语义修正——"冗余"实为**同病历双标注**

§1 的"23 条冗余"措辞偏粗，作者质询后复核定性（追加补记，不改上文）：

- **复核方法**：22 组逐组翻开比对——病历（condition）逐字节相同之外，比对 diseases/analyses/diagnoses。
- **定性**：22/22 组疾病列表完全相同（**无标签冲突**）；**0/22 组分析链完全相同**（20 组条数/内容实质
  差异，如 6 条含 1 分弱项 vs 4 条全 ≥2 分；最接近的一组 analyses 相似度 0.713）；1/22 组
  diagnoses 全同（rec-0103/0424）。结论：这批是**同病例过两遍标注管线的双标注**，不是复制粘贴。
- **三个真实用途**（训练侧剔除不变，理由见 §1/§5；素材未丢弃）：
  1. 多参考评测：同病例两套 gold 推理合并成更全证据集，依据覆盖轴的 recall 更公平；
  2. **MED-D 同输入偏好对**（第二来源，已入 PLAN §4）：输入完全一致、唯一变量=推理链，比 record 内
     score 对更干净；chosen 判定需人核（X4 前置覆盖此批）；
  3. 一致性探针：同病历两次生成应等价，部署期 self-consistency 校验的免费 ground truth。
- **落盘**：`split-manifest.json` 的 `removed_duplicates` 键由 `duplicate_groups` 取代（每组含
  hash/kept_id/dup_raw_indices/same_diseases/same_analyses/same_diagnoses/kind）。构建器同步改，
  复跑验证：**六个数据 fixture sha256 逐字节不变**（修正案不扰动既有产物），R0a/R0b/R0c 复验全绿。
