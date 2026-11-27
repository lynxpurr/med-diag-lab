# 数据出处与指纹（PROVENANCE）

> **公开版说明**：本仓库不分发 `data/raw/` 与各派生夹具——请运行 `bash data/fetch_raw.sh` 下载上游数据（脚本校验下方指纹），再按 README「快速开始」用 `tools/` 逐字节重建。随仓附带的唯一数据文件是 `data/medc/eval-neg.jsonl` 与 `eval-negn.jsonl`：本仓原创的合成无病体检文本，由大模型起草，**未经医生审核**。两份文件每条记录的 `draft_meta.reverse_probe` 停在起草时的“待跑”状态；探针实际已跑，结果见 `reports/med-b-sft.md`。 `draft_meta` 里的“人核”指 AI 助手代作者逐条核查，不是人工或医生审核；下文的“人工形态”指自由起草的 20 条，与由真实病例阴性化得到的 10 条相对，同样不是人手撰写。

## 源数据集

- 名称：`zhangmingme/medical-record-diagnosis-chinese`（病历诊断数据集）
- 平台：ModelScope（https://modelscope.cn/datasets/zhangmingme/medical-record-diagnosis-chinese）
- 许可：Apache License 2.0（**上传者声明**）
- 体量：1337 例 · 8.5MB（平台 API StorageSize=8550858）
- 下载方式（2026-09-27 实录）：

```bash
git clone --depth 1 https://www.modelscope.cn/datasets/zhangmingme/medical-record-diagnosis-chinese.git
# 仓库内为 LFS 指针，本机无 git-lfs，走 resolve URL 直下：
curl -sL -o medical-record-diagnosis.json \
  "https://modelscope.cn/datasets/zhangmingme/medical-record-diagnosis-chinese/resolve/master/medical-record-diagnosis.json"
curl -sL -o medical-record-diagnosis-sft.jsonl \
  "https://modelscope.cn/datasets/zhangmingme/medical-record-diagnosis-chinese/resolve/master/medical-record-diagnosis-sft.jsonl"
```

## 指纹（data/raw/SHA256SUMS）

| 文件 | sha256 | 与 ModelScope LFS oid 对账 |
|---|---|---|
| medical-record-diagnosis.json | `42e4f2ad4094505ac7f1250f9b37aef30065aa5a9924ea65604630c34fa823ff` | **逐位一致** ✓ |
| medical-record-diagnosis-sft.jsonl | `c841f4864eb0678ecf8f6dc57c97717e17827ae752d2343325b0dde667cf0359` | **逐位一致** ✓ |

（oid 出处：clone 仓库内 LFS 指针文件 `medical-record-diagnosis.json` / `-sft.jsonl` 原文）

## 出处限定（D-02）

- 上传者为个人（zhangmingme），内容风格为教科书/病例分析题式典型病例（含霍乱/伤寒/麻疹/新生儿疾病等），
  **原始出处不可独立核实**——网络检索（2026-09-27）未找到论文或上游仓库。
- 使用边界：**限学习/研究用途**；产物不做任何临床可用性声明（PLAN X5）。

## 衍生数据

- `data/med0/`：MED-0 构建产物（去重/切分/三形态），构建器 `tools/build_dataset.py`，seed 20260925，
  判决与统计见 `reports/med-0-data.md`。
- `data/medc/`（MED-C，2026-09-29）：素材正式化 + 合成增强两类。
  - **素材正式化（评测材料）**：`eval-neg.jsonl`（neg-001~020，人工形态 A10/B6/C4）与
    `eval-negn.jsonl`（negn-001~010，阴性化对照）——git mv 自 research/ 草稿（内容未动，
    2026-09-28 起草）。出处与验证：GLM-5.3 模型起草/机械导出（作者非医学专家，授权联网
    查询代行数据集审阅）+ 数值锚定 WS/T 404/405-2012 + TSH 0.27-4.2 mIU/L 通行参考区间
    （260 项逐断言）+ 反面探针 30/30（14B 零样本与 MED-B 冠军两测点，判病即修条回路）。
    评测器映射：正式键 neg/negn（旧键 neg-draft/negn-draft 保留为别名）。许可随源数据集
    Apache-2.0 边界 + 本仓 X5（限学习/研究，不做临床可用性声明）。
  - **合成增强（训练材料）**：`aug-garbled/minimal/neg.jsonl` 各 36 条 + 臂合并集
    `swift-train-c{g,m,n}.jsonl`（f1 1154 原序 + aug），构建器 `tools/augment_med_c.py`
    （augment.py 谱系，seed 20260925，确定性选样）。变体源全部 ∈ f1 train（评测集一字
    不入）；无病臂为新生成（人口学锚定 train + 健康模板 + WS/T 数值锚 + 现症阳性扫描 +
    与 30 条评测素材三层文本防线：主诉非逐字同 ∧ 全文 ratio<0.85 ∧ 叙事块 ratio<0.80），
    造数质量门 = probe-negt.jsonl 冠军反面探针（读数见 `reports/med-c-robust.md`）。
    选样/配额/断言汇总见 `aug-manifest.json`。
- `data/medd/`（MED-D，2026-09-29）：DPO 偏好对 300（A 源 score 对 294 + B 源双标注 6），
  构建器 `tools/build_dpo_pairs.py`（seed 20260925）；对拍 `tools/check_dpo_med.py` 全绿
  （含出口形态扫描与组装保真两道本仓增量门）。细则见 `data/medd/PROVENANCE.md`。
