#!/usr/bin/env python3
"""MED-B 前置模板对拍：官方 jinja 渲染 vs swift 4.5.3 qwen3 系（check_template_d1 谱系第五次复用）.

d1 的拷贝，三处不同（PLAN §4 MED-B 开工前置，2026-09-28 补）：
  1. 数据源换本仓 data/med0/swift-train-f1.jsonl（官方式三消息行），抽 2 条 =
     首条 + 首条多疾病记录（目标形态多样性和单疾病各覆盖一）；
  2. 模型 = $MODELS_DIR/<模型名>，模型名缺省 MED-B 主档 Qwen3-14B（第 2 参数可换，
     如 Qwen3-1.7B 快筛档）；
  3. 实效口径不变：get_template 带 loss_scale="default+ignore_empty_think"
     （swift sft 对 hybrid-thinking 模板自动追加的实效 loss_scale；空 think 块
     在官方 jinja 的生成前缀侧 vs swift 的内容侧——裸 encode 显 4 token 假红，
     d1 实证，此处不复述）；
  4. max_length=2048（d1 用 1024）：2026-09-28 实测 F1 全 1154 行 token 长度
     1096~1725（14B tokenizer），1024 会全量截断——训练 max_length 必用 2048。

用法（ref 先跑；换档加第 2 参数。环境变量 MODELS_DIR = Qwen3 模型根目录）：
    python tools/check_template_med.py ref              # 只需 transformers
    python tools/check_template_med.py swift            # 需 ms-swift
    python tools/check_template_med.py ref Qwen3-1.7B   # MED-B 快筛档
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL_NAME = sys.argv[2] if len(sys.argv) > 2 else "Qwen3-14B"
MODELS_DIR = os.environ.get("MODELS_DIR")  # Qwen3 模型根目录（README「快速开始」）
assert MODELS_DIR, "请先 export MODELS_DIR=<Qwen3 模型根目录>"
MODEL = str(Path(MODELS_DIR) / MODEL_NAME)
TRAIN_F1 = REPO / "data" / "med0" / "swift-train-f1.jsonl"
REF_JSON = REPO / "output" / "med-b" / f"template-ref-{MODEL_NAME}.json"  # ref → swift 两步之间传递渲染结果


def _pick_rows() -> list[dict]:
    rows = [json.loads(ln) for ln in TRAIN_F1.read_text(encoding="utf-8").splitlines() if ln.strip()]
    first_multi = next(r for r in rows if len(json.loads(json.dumps(r))["messages"][-1]["content"].split("<diseases>")[1].split("</diseases>")[0].strip().rstrip(";").split(";")) >= 2)
    return [rows[0], first_multi]


def _ref(tok) -> dict:
    """官方 jinja 同式渲染：prompt=前 2 消息（system+user）加生成头，full=全部 3 消息."""
    out = []
    for row in _pick_rows():
        msgs = row["messages"]
        prompt = tok.apply_chat_template(
            msgs[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        assert full.startswith(prompt), "jinja 模板不连续（不应发生）"
        enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
        cut = next((i for i, (s, _) in enumerate(enc["offset_mapping"]) if s >= len(prompt)), len(enc["input_ids"]))
        out.append({"id": row["id"], "form": row.get("form"), "prompt": prompt, "full": full, "cut": cut})
    return out


def main_ref() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    REF_JSON.parent.mkdir(parents=True, exist_ok=True)
    REF_JSON.write_text(json.dumps(_ref(tok), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ref 渲染完成 → {REF_JSON}（2 条：单疾病 + 多疾病；model={MODEL}）")
    return 0


def main_swift() -> int:
    from swift.model import get_processor
    from swift.template import get_template

    processor = get_processor(MODEL)
    tok = processor if hasattr(processor, "decode") else processor.tokenizer  # 4.5.3: processor 即 tokenizer
    refs = json.loads(REF_JSON.read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in _pick_rows()}

    candidates = [
        ("qwen3 + enable_thinking=False", dict(template_type="qwen3", enable_thinking=False,
                                               loss_scale="default+ignore_empty_think")),
        ("qwen3_nothinking", dict(template_type="qwen3_nothinking",
                                  loss_scale="default+ignore_empty_think")),
    ]
    verdict = {}
    for name, kw in candidates:
        template = get_template(processor, max_length=2048, **kw)
        template.init_processor(processor)
        template.set_mode("train")
        ok_text = ok_mask = True
        detail = []
        for ref in refs:
            row = rows[ref["id"]]
            enc = template.encode({"messages": row["messages"]})
            text = tok.decode(enc["input_ids"], skip_special_tokens=False)
            if text != ref["full"]:
                ok_text = False
                i = next((k for k, (a, b) in enumerate(zip(text, ref["full"])) if a != b), min(len(text), len(ref["full"])))
                detail.append(f"{ref['id']}: 文本首个差异@{i} swift={text[max(0,i-30):i+30]!r} ref={ref['full'][max(0,i-30):i+30]!r}")
            boundary_swift = sum(1 for v in enc["labels"] if v == -100)
            if boundary_swift != ref["cut"]:
                ok_mask = False
                detail.append(f"{ref['id']}: 掩码边界 swift={boundary_swift} ref={ref['cut']}")
        verdict[name] = (ok_text, ok_mask, detail)
        print(f"\n== {name} ==")
        print(f"  文本逐字相等: {ok_text} | 掩码边界相等: {ok_mask}")
        for d in detail:
            print(f"  - {d}")
    winners = [n for n, (t, m, _) in verdict.items() if t and m]
    print(f"\n结论：{'✓ ' + '；'.join(winners) if winners else '✗ 无候选全对上——需进一步适配'}")
    return 0 if winners else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"ref": main_ref, "swift": main_swift}.get(mode, lambda: print(__doc__) or 2)())
