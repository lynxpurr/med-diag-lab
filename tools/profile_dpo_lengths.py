#!/usr/bin/env python3
"""MED-D D2 显存适配：A 源资格池全量侧长离线测量 → data/medd/length-profile.json.

背景：14B QLoRA DPO 的 micro-batch = chosen+rejected 拼批 [2, L]（L=两侧最长），
bnb 4bit 反向去量化缓冲 × 边界激活在最长的对上 OOM（三次实测缺口 ~100-400MiB，
实录 output/med-d/14b-dpo-s1-oom*-0929）。12300 需求线是 SFT 单序列 2048 的口径，
DPO 拼批天然 +46% 激活——需要给偏好对加"长度资格"：max(侧长) ≤ CAP，超限对由
同资格池等量短对替换（保 300 对预算与步数剂量不动，组成变化入 manifest 记账）。

测量口径与 check_dpo_med.py 同：rlhf 模式 template encode 后的 chosen/rejected input_ids 长度
（含 system/user/空 think 块全部模板 token）。CPU-only，~2min。
用法：python tools/profile_dpo_lengths.py   （需 ms-swift；环境变量 MODELS_DIR = Qwen3 模型根目录）
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = os.environ.get("MODELS_DIR")  # Qwen3 模型根目录（README「快速开始」）
assert MODELS_DIR, "请先 export MODELS_DIR=<Qwen3 模型根目录>"
MODEL = str(Path(MODELS_DIR) / "Qwen3-14B")


def main() -> int:
    raw = json.loads((REPO / "data" / "raw" / "medical-record-diagnosis.json").read_text(encoding="utf-8"))
    sft = [json.loads(ln) for ln in (REPO / "data" / "raw" / "medical-record-diagnosis-sft.jsonl")
           .read_text(encoding="utf-8").splitlines() if ln.strip()]
    manifest = json.loads((REPO / "data" / "med0" / "split-manifest.json").read_text(encoding="utf-8"))
    system_prompt = sft[0]["messages"][0]["content"]

    from swift.model import get_processor
    from swift.template import get_template
    processor = get_processor(MODEL)
    template = get_template(processor, max_length=2048, template_type="qwen3",
                            enable_thinking=False, loss_scale="last_round")
    template.init_processor(processor)
    template.set_mode("rlhf")
    from swift.template.template_inputs import TemplateInputs

    def target(diseases: list[str], descs: list[str]) -> str:
        return "<diseases>\n" + ";".join(diseases) + ";\n</diseases>" + \
            "\n" + "\n".join(f"{i}. {d}" for i, d in enumerate(descs, 1))

    profile = {}
    for rid in sorted(manifest["splits"]["train"]):
        i = int(rid.split("-")[1])
        rec = raw[i]
        scores = [a["score"] for a in rec["analyses"]]
        if not (3.0 in scores and any(s <= 2 for s in scores)):
            continue  # 非 A 源资格（1043 池同口径）
        row = {"messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": sft[i]["messages"][1]["content"]},
                            {"role": "assistant", "content":
                             target(rec["diseases"], [a["description"] for a in rec["analyses"] if a["score"] == 3])}],
               "rejected_response": target(rec["diseases"],
                                           [a["description"] for a in rec["analyses"] if a["score"] <= 2])}
        enc = template.encode(TemplateInputs.from_dict(row))
        profile[rid] = {"len_chosen": len(enc["chosen_input_ids"]),
                        "len_rejected": len(enc["rejected_input_ids"])}

    assert len(profile) == 1043, f"资格池 {len(profile)} ≠ 1043"
    out = REPO / "data" / "medd" / "length-profile.json"
    out.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    worst = sorted((max(v["len_chosen"], v["len_rejected"]) for v in profile.values()), reverse=True)
    sides = [max(v["len_chosen"], v["len_rejected"]) for v in profile.values()]
    print(f"[profile] 1043 资格记录 → {out}")
    print(f"[profile] max(侧长) 分布：p50={sorted(sides)[521]} p90={sorted(sides)[938]} "
          f"p95={sorted(sides)[990]} max={worst[0]}")
    for cap in (1500, 1450, 1400, 1350, 1300, 1200):
        keep = sum(1 for s in sides if s <= cap)
        print(f"[profile] cap={cap}: 资格保留 {keep}/1043")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
