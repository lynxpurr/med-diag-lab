# data/medd/ — MED-D DPO 偏好对（2026-09-29）

## 文件

| 文件 | 内容 |
|---|---|
| `dpo-pairs.jsonl` | 300 对偏好数据（A 源 294 + B 源 6），swift rlhf dpo 格式：`messages`(system+user+assistant=chosen 裸) + `rejected_response`(裸) |
| `manifest.json` | 构建 audit：seed/抽样法/来源分解/逐对项数/泄题断言/sha256 |

构建器：`tools/build_dpo_pairs.py`（seed 20260925 确定性，复跑逐字节同产）。
对拍器：`tools/check_dpo_med.py`（G-4 工程检查单拷贝改造 + 出口形态/组装保真两道本仓增量，
2026-09-29 全绿：encode 300/300、think 块公共前缀、长度 p95 ≤1303 < 门、出口违约 0/0、
verbatim 保真全过）。

**长度资格（2026-09-30 D2 显存适配追加）**：A 源资格 ∧ max(侧长) ≤ 1350 tok——14B QLoRA
DPO 拼批 [2,L] 在 16GB 共享卡三次 OOM 后由截断探针定案（seq 伸缩 ~2.7MB/tok 实测，
配方峰值 12.39GiB）；资格池 1043 保留 973，超限对同池等量替换（保 300 预算与剂量阶梯
不动）。侧长离线实测 = `tools/profile_dpo_lengths.py` → `length-profile.json`（1043 全量）；
B 源 6 对整链实测 1142-1317 全保留。详见 `reports/med-d-dpo.md` §2。

## 两来源（判定口径 = X4 工作表 `research/2026-09-28-x4-pairs-worksheet.md`）

- **A 源 294 对**（`source: a-score`）：train split 内同病历 record，chosen = score=3 项集 vs
  rejected = score≤2 项集（组装形态 `build_target` 同构：英文分号 diseases 块 + 编号依据链）。
  X4 模型初判 score 信号可信（Y 29/30 = 96.7% ≥ 80% 线）→ 直入预算不加人核层。
  抽样 = `Random(20260925).sample(sorted(资格池 1043 − B 源重合), 294)`——与 X4 30 对同法同 seed。
- **B 源 6 对**（`source: b-dup`）：同病历双标注整套链互换（manifest `duplicate_groups`），
  X4 §B 人核代行 K3+D3：K（rec-0154/0427/0823）chosen=kept 链；D（rec-0740/0793/0923）
  chosen=dup 链。余 14 组 T（语义同构）不作对。

## 防线

- **泄题**：pair 全部 ∈ train split（1154）；val-reg/test-ood/rob_base 断言零交集（构建器内硬门）。
- **出口契约**（MED-C 教训前置）：两侧 diseases 块同源官方 diseases（英文分号），chosen 侧
  违约出口（"健康/正常"类）= 废对硬门——对拍实测 0。
- 源数据质量归 MED-0 审计域；本构建只做组装保真（依据项 verbatim ∈ 官方 analyses）。

## 许可

随源数据集（Apache-2.0 边界）+ 本仓 X5：限学习/研究用途，不做任何临床可用性声明。
