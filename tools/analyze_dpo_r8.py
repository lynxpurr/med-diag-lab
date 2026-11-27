#!/usr/bin/env python3
"""MED-D R8 离线读数：负侧消退率代理 + 出口词形态对比（champion vs DPO 三测点）.

R8 预注册（PLAN §4）：起点 logps 已分离（实证见 s3 logging：chosen -19.6 vs rejected -55.5），
评估看负侧行为消退。本工具给两个可计算代理：
  ① 依据条目结构（val-reg）：rejected 优势形态 = 少量概括条目（中位 1 项）、chosen = 3-4 项
     递进链（X4 附观察）——若 DPO 打压负侧生效，响应的条目数分布应相对冠军上移
     （结构完整性增强 = "条目粒度+证据类型递进"的 chosen 形态被强化）。
  ② 出口词形态（neg/negn）：违约出口（"健康/正常"字面）计数——负侧消退的出口层读数
     （MED-C 教训：cg/cm 臂曾漂移 5/7；DPO 对比损失恰是打压坏出口的机制）。

输入：既有证据 jsonl（swift infer 保序，无 id 字段——与 eval 集按行 zip）。
用法：python3 tools/analyze_dpo_r8.py            # val-reg 条目分布 + neg/negn 出口扫描
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DISEASES_BLOCK = re.compile(r"<diseases>\n(.*?)\n</diseases>", re.S)
ITEM = re.compile(r"^(\d+)\. ", re.M)
VIOLATION = ("健康", "正常")

MODELS = {  # tag → (证据目录, val-reg 文件名)
    "champion": ("output/med-b", "sft-14b-f1-val-reg.jsonl"),
    "dpo-s1": ("output/med-d", "dpo-s1-val-reg.jsonl"),
    "dpo-s3": ("output/med-d", "dpo-s3-val-reg.jsonl"),
    "dpo-s9": ("output/med-d", "dpo-s9-val-reg.jsonl"),
}
NEG_SETS = {  # 集名 → [(目录, 文件名)]，champion 用 MED-B 时代 neg-draft 同内容键
    "neg": {"champion": ("output/med-b", "sft-14b-f1-neg-draft.jsonl"),
            "dpo-s1": ("output/med-d", "dpo-s1-neg.jsonl"),
            "dpo-s3": ("output/med-d", "dpo-s3-neg.jsonl"),
            "dpo-s9": ("output/med-d", "dpo-s9-neg.jsonl")},
    "negn": {"champion": ("output/med-b", "sft-14b-f1-negn-draft.jsonl"),
             "dpo-s1": ("output/med-d", "dpo-s1-negn.jsonl"),
             "dpo-s3": ("output/med-d", "dpo-s3-negn.jsonl"),
             "dpo-s9": ("output/med-d", "dpo-s9-negn.jsonl")},
}


def rows_of(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [r.get("response") or (r.get("messages") or [{}])[-1].get("content", "")
            for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r]


def item_count(resp: str) -> int:
    body = resp.split("</diseases>", 1)[1] if "</diseases>" in resp else resp
    return len(ITEM.findall(body))


def exit_violations(resp: str) -> int:
    blocks = DISEASES_BLOCK.findall(resp)
    if len(blocks) != 1:
        return 1 if blocks else 0  # 块缺失/多重都算形态异常
    return sum(1 for e in blocks[0].split(";") if e and any(v in e for v in VIOLATION)
               and e not in ("无", "数据异常", "缺少关键诊断信息"))


def main() -> int:
    print("== ① val-reg 依据条目数分布（结构完整性代理；rejected 形态中位 1 项 / chosen 3-4 项）==")
    base_counts = None
    for tag, (d, f) in MODELS.items():
        rows = rows_of(REPO / d / f)
        if not rows:
            print(f"  {tag:9s} 证据缺（跳过）")
            continue
        counts = [item_count(r) for r in rows]
        if tag == "champion":
            base_counts = counts
        line = (f"  {tag:9s} n={len(rows)} 均值 {statistics.mean(counts):.2f} 中位 "
                f"{statistics.median(counts):.1f} p10/p90 {sorted(counts)[len(counts)//10]}/"
                f"{sorted(counts)[len(counts)*9//10]}")
        if base_counts and len(base_counts) == len(counts):
            delta = [c - b for c, b in zip(counts, base_counts)]
            up = sum(1 for x in delta if x > 0)
            dn = sum(1 for x in delta if x < 0)
            line += f" · vs 冠军：条目↑{up} ↓{dn} 净{sum(delta):+d}"
        print(line)

    print("\n== ② neg/negn 出口词形态（违约出口 = '健康/正常'类字面；MED-C cg/cm 曾 5/7）==")
    for set_name, mp in NEG_SETS.items():
        for tag, (d, f) in mp.items():
            rows = rows_of(REPO / d / f)
            if not rows:
                continue
            viol = sum(exit_violations(r) for r in rows)
            print(f"  {set_name:4s} {tag:9s} n={len(rows)} 违约出口 {viol} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
