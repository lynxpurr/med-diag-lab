# SPEC-INDEX — 规格索引（路由器）

> **公开版说明**：本文件按归档时的原样保留，只清理了本机环境信息与私有仓标识。文中的 `output/…` 指运行产物目录，公开版只收录它的数值证据导出 `evidence/`；“skill”指 AI 助手用的操作手册，未随公开版发布；章节号指写作当时的版本。

> 建立：2026-09-28（D-07）· **本件只路由不双写**（同 skill 哲学）：每类规格的**事实源**
> 在下表所指之处，本件不复述内容——规格变了本件不用改，除非事实源搬家。
> 为什么不设 `specs/` 目录：本仓规格以**可执行形态优先**（gold 门/断言/快照，机器执行、
> 当场抓虫），文字 spec 是第二事实源，拆得越细漂移面越大；文档账本已满编（PLAN/HANDOFF/
> playbook/lessons/map/reports/research 七层各司其职）。拆 `specs/` 的触发条件：PLAN 膨胀
> 到不可维护，或 MED-E 后项目延长/换手——届时按姊妹域实验仓的模板（其 `specs/` 目录）拆并记决策。

| 规格类型 | 事实源 | 可执行验证 | 变更路径 |
|---|---|---|---|
| 实验 spec（预注册判决 R0~R9） | `PLAN.md` §4 各波条目 | 判决按原文逐条落卷（报告核对） | **判决层冻结**；设计层走校准块（改后续波设计，PLAN 记变更） |
| 判分 spec（五轴 + 轴⑥口径 v1.1+） | `tools/run_eval_med.py` 模块 docstring | **gold 自校准门**（改口径必复跑，全满分才许跑模型） | 口径修正记 docstring 修正号 + 报告；冻结跨波基线须记 PLAN |
| 规范化映射（SYNONYMS） | `tools/normalize.py` | gold 门间接覆盖 | **追加制**（带样例 id 与日期），改必记 PLAN 变更 |
| 数据 spec（切分/形态/契约） | `tools/build_dataset.py`（断言即规格） | 六 fixture sha256 复跑不变 + R0a/b/c 断言 | 复跑验证后 PROVENANCE/manifest 同步 |
| 数据出处与指纹 | `data/PROVENANCE.md` + `data/raw/SHA256SUMS` | `sha256sum -c` | 外部新数据须先过 PROVENANCE + 审计（红线） |
| 任务契约（system prompt） | `data/raw/medical-record-diagnosis-sft.jsonl` 首行 | 评测器/训练器**运行时现取**（不复制粘贴） | 官方原文，不可变 |
| 训练配置 spec（超参冻结） | `tools/run_med_b.sh`（从底座三臂）+ `tools/run_med_c.sh`（冠军续训形态：--adapters 挂 ckpt + 末步门动态计算）+ `tools/run_med_d.sh`（DPO 形态：双 adapters 同指冠军 · 步数臂 --max_steps · 拼批口径显存线 12900） | 每 run `args.json` 快照核对（max_length/loss_scale 实效口径/quant/seed/续训的 adapters 落档/DPO 加 rlhf_type·lr 5e-6·β） | 臂间唯一变量=数据形态；改超参=破三臂可比性，须记 PLAN |
| 偏好数据 spec（MED-D 对） | `data/medd/dpo-pairs.jsonl` + `manifest.json`（两来源/长度资格 cap1350/泄题断言） | `tools/check_dpo_med.py` 对拍全绿（encode/公共前缀/出口扫描/组装保真）；`tools/build_dpo_pairs.py` 复跑确定性 | 重建须过对拍再入训练；长度资格变更须重过显存实证（探针法） |
| 无病/探针素材规格（已正式化） | `data/medc/eval-neg.jsonl` + `eval-negn.jsonl` 每条 `draft_meta` 字段（2026-09-29 自 research/ git mv，旧路径失效） | 数值锚定 WS/T 404/405 断言 + 反面探针（模型判病即修条）+ gold 门（评测器键 neg/negn） | 修正走 PROVENANCE 追加 |
| 合成增强素材规格（MED-C 造数） | `data/medc/aug-*.jsonl` + `aug-manifest*.json`（选样/配额/断言汇总） | `tools/augment_med_c.py` fidelity 断言全绿才落盘（评测器复判门 + 三层泄题防线 + 探针集 probe-negt） | 造数器即规格（augment.py 谱系）；重造须删旧产物重跑（证据只增不改） |
| 操作流程 spec | `.zcode/skills/med-diag-finetune/SKILL.md`（路由器；私有工作副本的 agent 操作手册，**未随公开版发布**——操作要点见 `tools/` 脚本头注与 playbook §6） | 开工序逐项可执行 | 资产坐标原地更新；方法论进 playbook 不进 skill |
| 域结论（读数/门槛/失败形态） | `notes/map.md`（罗盘，波次戳追加制） | 读数锚定 reports 证据文件 | 新批声明取代关系，不原地改写 |
| 方法论细则 | `notes/playbook.md`（27 条+） | —— | 追加制，带日期修正条 |
| 过程教训 | `notes/lessons.md` | —— | 追加制 |
| 波次证据 | `reports/`（读数明细）+ `output/`（原始产物；公开版只收录其数值证据导出 `evidence/`） | 证据只增不改（assert not exists） | 修正走追加补记，不回改 |

**使用规则**：找任何规格的定义 → 先查本表到事实源；发现本表指向失效（文件搬家/改名）→
修本表并记 PLAN 变更一行；两处表述冲突 → 以本表所指事实源为准，本表永远不是事实源。
