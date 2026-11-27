# evidence/ — 数值证据

本目录是实验运行产物目录（文档中写作 `output/`）里数值证据的导出，路径与之一一对应。读数与运行时产物一致，只做了两处改动：评分文件去掉了每条记录的模型回复正文（`resp_head`）；参数快照和训练日志里的本机路径改写成了仓库相对路径（模型目录写作 `$MODELS_DIR/`）。模型回复全文、GPU 采样日志、控制台输出和模型权重没有导出。

## 文件类型

| 文件 | 内容 |
|---|---|
| `<波次>/score-<tag>-<评测集>.json` | 评测器对一次推理的逐条打分与汇总（`aggregate`）。字段含义见 `tools/run_eval_med.py` 的模块说明 |
| `<波次>/<run>/args.json` | ms-swift 训练参数快照 |
| `<波次>/<run>/logging.jsonl` | 逐步训练日志：loss、梯度范数、显存 `memory(GiB)`、速度 `train_speed(s/it)`；DPO 另有 rewards 与 logps。最后两行是运行汇总：耗时与吞吐，以及内嵌完整训练历史的 `log_history` |
| `med-e/stratify.json` | 长尾分层读数，由 `tools/med_e_stratify.py` 从评分文件算出 |

目录名带 `-oom-` 的是显存不足的失败尝试，只留下参数快照和一行汇总日志。

## tag 与配方的对应

| tag | 配方 |
|---|---|
| `0.6b` `1.7b` `4b` `8b` `14b` | Qwen3 各档零样本（`4b` 是 Qwen3-4B-Instruct-2507） |
| `zs-14b` | Qwen3-14B 零样本在无病集上的探针 |
| `sft-1p7b-f1` | 1.7B 快筛：SFT，官方目标形态 |
| `sft-14b-f1` `sft-14b-f2` `sft-14b-f3c` | 14B SFT 三臂：官方式 / 只保留 score≥2 的分析 / 凝练依据 |
| `sft-medc-cg` `sft-medc-cm` `sft-medc-cn` `sft-medc-cnlow` | 在 SFT 冠军上续训的增强臂：乱码 / 极简 / 无病 / 低配比无病（文档里写作 cn-low） |
| `dpo-s1` `dpo-s3` `dpo-s9` | 在 SFT 冠军上做 DPO，步数 1 / 3 / 9 |
| `probe-champ-f1` `probe-champ-f1-r2` | 合成训练素材的造数质量探针（`negt` 集），不入判卷 |

评测集后缀里的 `neg-draft` / `negn-draft` 是 `neg` / `negn` 的旧名，内容相同。

## 从证据重新算出报告里的表

```bash
python3 tools/run_eval_med.py table --out evidence/med-b      # 任一波次的主表
EVIDENCE_ROOT=evidence python3 tools/med_e_stratify.py        # 长尾分层 → output/med-e/stratify.json
```

报告里的显存峰值和每步耗时，取自对应 `logging.jsonl` 的 `memory(GiB)` 与 `train_speed(s/it)` 字段的最大值和中位数。
