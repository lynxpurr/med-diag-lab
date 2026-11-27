#!/usr/bin/env python3
"""MED-D D1 前置对拍：300 偏好对在 swift 4.5.3 qwen3 模板下的 encode 实证 + 本仓增量扫描.

拷贝改造自上游方法学仓的 check_dpo_g4.py（G-4 工程检查单），数据换 data/medd/dpo-pairs.jsonl。
验证五点（实效对拍原则——裸 encode 会假红，按 rlhf 训练实参 encode）：
  1. rlhf 路径通：每行 encode 产出 chosen_input_ids / rejected_input_ids；
  2. 空 think 块落公共前缀（add_non_thinking_prefix 对裸 chosen/rejected 对称加前缀，
     DPO 信号不被 think 差异污染），两侧各恰好一个 '</think>'；
  3. 长度门：两侧 token 均 < 2048（=训练 max_length；截断行会恰为 2048 被捕获）；
  4. 出口形态扫描（本仓增量·MED-C 教训）：chosen/rejected 的 <diseases> 块出口契约
     ——英文分号分隔、无中文分号、无"健康/正常"类字面违约出口（chosen 侧违约即废对，
     rejected 侧漂移形态入账——DPO 对比损失恰是打压坏出口的机制）；
  5. 组装保真扫描（数值一致性的本仓化）：两侧编号依据项逐条 verbatim ∈ 源 record 官方
     analyses 项集——组装零损（数值随文本保真，源数据质量归 MED-0 审计域）。

用法：python tools/check_dpo_med.py   （需 ms-swift；环境变量 MODELS_DIR = Qwen3 模型根目录）
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = os.environ.get("MODELS_DIR")  # Qwen3 模型根目录（README「快速开始」）
assert MODELS_DIR, "请先 export MODELS_DIR=<Qwen3 模型根目录>"
MODEL = str(Path(MODELS_DIR) / "Qwen3-14B")
MAXLEN = 2048  # 训练 max_length（F1 序列 1096~1725 token 实测 → 必用 2048）

VIOLATION_EXITS = ("健康", "正常")  # MED-C 出口词漂移形态（语义对·字面违约）
DISEASES_BLOCK = re.compile(r"<diseases>\n(.*?)\n</diseases>", re.S)
ITEM = re.compile(r"^(\d+)\. (.+)$", re.M)


def scan_exits(text: str, side: str, rid: str, ledger: dict) -> list[str]:
    """出口契约扫描：返回违约描述列表（chosen 侧非空 = 废对）。"""
    problems = []
    blocks = DISEASES_BLOCK.findall(text)
    if len(blocks) != 1 or not text.startswith("<diseases>"):
        return [f"{rid}[{side}] diseases 块形态异常（{len(blocks)} 个）"]
    entries = [e for e in blocks[0].split(";") if e]
    if not blocks[0].endswith(";"):
        problems.append(f"{rid}[{side}] 块尾缺英文分号")
    if "；" in blocks[0]:
        problems.append(f"{rid}[{side}] 中文分号违约")
    for e in entries:
        if any(v in e for v in VIOLATION_EXITS) and e not in ("无", "数据异常", "缺少关键诊断信息"):
            problems.append(f"{rid}[{side}] 违约出口词：{e!r}")
            ledger[f"{side}:违约出口"] = ledger.get(f"{side}:违约出口", 0) + 1
    return problems


def main() -> int:
    rows = [json.loads(ln) for ln in (REPO / "data" / "medd" / "dpo-pairs.jsonl")
            .read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 300, f"行数 {len(rows)} ≠ 300"
    raw = json.loads((REPO / "data" / "raw" / "medical-record-diagnosis.json").read_text(encoding="utf-8"))
    manifest = json.load(open(REPO / "data" / "med0" / "split-manifest.json"))
    gmap = {g["kept_id"]: g for g in manifest["duplicate_groups"]}

    from swift.model import get_processor
    from swift.template import get_template
    processor = get_processor(MODEL)
    tok = processor if hasattr(processor, "decode") else processor.tokenizer
    template = get_template(processor, max_length=MAXLEN, template_type="qwen3",
                            enable_thinking=False, loss_scale="last_round")
    template.init_processor(processor)
    template.set_mode("rlhf")
    from swift.template.template_inputs import TemplateInputs

    ledger: dict = {"chosen:违约出口": 0, "rejected:违约出口": 0}
    fails: list[str] = []
    len_c, len_r = [], []
    for r in rows:
        rid = r["id"]
        # ---- ①②③ rlhf encode：路径/公共前缀/长度门 ----
        encoded = template.encode(TemplateInputs.from_dict(r))
        ci, ri = encoded["chosen_input_ids"], encoded["rejected_input_ids"]
        len_c.append(len(ci)), len_r.append(len(ri))
        for tag, ids in (("chosen", ci), ("rejected", ri)):
            if len(ids) >= MAXLEN:
                fails.append(f"{rid}[{tag}] 长度 {len(ids)} ≥ {MAXLEN}（截断风险）")
        ct, rt = tok.decode(ci, skip_special_tokens=False), tok.decode(ri, skip_special_tokens=False)
        if ct.count("</think>") != 1 or rt.count("</think>") != 1:
            fails.append(f"{rid} think 块非单次")
            continue
        common = 0
        for a, b in zip(ci, ri):
            if a != b:
                break
            common += 1
        if "</think>" not in tok.decode(ci[:common], skip_special_tokens=False):
            fails.append(f"{rid} think 块未落公共前缀（两侧不对称）")

        chosen_text, rejected_text = r["messages"][2]["content"], r["rejected_response"]
        # ---- ④ 出口形态扫描 ----
        fails += scan_exits(chosen_text, "chosen", rid, ledger)
        scan_exits(rejected_text, "rejected", rid, ledger)  # 负侧漂移只入账不废对
        if DISEASES_BLOCK.findall(chosen_text) != DISEASES_BLOCK.findall(rejected_text):
            fails.append(f"{rid} 两侧 diseases 块不同（设计不变量：偏好信号只在依据链）")

        # ---- ⑤ 组装保真：依据项 verbatim ∈ 源官方链 ----
        if r["source"] == "a-score":
            src_descs = {a["description"] for a in raw[int(rid.split("-")[1])]["analyses"]}
        else:
            g = gmap[rid]
            src_descs = {a["description"] for a in raw[g["kept_raw_index"]]["analyses"]}
            src_descs |= {a["description"] for a in raw[g["dup_raw_indices"][0]]["analyses"]}
        for side, text in (("chosen", chosen_text), ("rejected", rejected_text)):
            items = ITEM.findall(text.split("</diseases>", 1)[1])
            if [n for n, _ in items] != [str(i) for i in range(1, len(items) + 1)]:
                fails.append(f"{rid}[{side}] 编号不连续")
            for _, desc in items:
                if desc not in src_descs:
                    fails.append(f"{rid}[{side}] 依据项非源官方文本（组装失真）：{desc[:40]}…")

    len_c.sort(), len_r.sort()
    print(f"[1] rlhf encode 300/300 通过；chosen len p50/p95/max = "
          f"{len_c[150]}/{len_c[284]}/{len_c[-1]}，rejected = "
          f"{len_r[150]}/{len_r[284]}/{len_r[-1]}（门 <{MAXLEN}）")
    print(f"[2] think 块公共前缀 + 单次性：{'全过' if not any('think' in f for f in fails) else '见失败清单'}")
    print(f"[3] 出口形态扫描：chosen 违约 {ledger['chosen:违约出口']}（须 0）· "
          f"rejected 违约 {ledger['rejected:违约出口']}（入账，可 >0）")
    print(f"[4] 组装保真（verbatim 项归属）：{'全过' if not any('失真' in f for f in fails) else '见失败清单'}")
    if fails:
        print(f"\n失败 {len(fails)} 条：")
        for f in fails[:20]:
            print("  ✗", f)
        return 1
    print("\n结论：300 对全绿——格式/公共前缀/长度/出口形态/组装保真均过，可入 DPO 训练")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
