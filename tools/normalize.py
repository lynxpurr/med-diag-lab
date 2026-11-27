#!/usr/bin/env python3
"""诊断结论规范化（评测口径资产——X3 风险的缓解件）.

用途：疾病 F1 计算前，把模型输出与 gold 的疾病名拉到同一可比形态。
纪律（PLAN D-04）：映射表**追加制**——MED-A 评测开发期按错误分析逐条追加，每次追加在
SYNONYMS 变更注释里记日期与来源样例；改它必须记 PLAN 变更记录，否则 F1 读数跨波不可比。

只做保守规范化：空白/全半角标点/结尾分号/常见前缀后缀。不做激进同义词合并（如
"门脉性肝硬变"→"肝硬化" 这类术语合并必须走 SYNONYMS 显式登记）。
"""
from __future__ import annotations

import re

# 追加制同义词表：规范名 <- 变体（评测开发期按 val-reg 错误分析填充）
SYNONYMS: dict[str, str] = {
    # 2026-09-27 MED-0：起点为空。首个追加者请附样例 id 与双方原文。
    # 2026-09-28 MED-A 首批（近失误分析筛选：仅登记同一实体的命名变体，亚型/粒度差不入表）：
    "慢性阻塞性肺病": "慢性阻塞性肺疾病",  # 样例 0.6b val-reg rec-0183
    "类风湿性关节炎": "类风湿关节炎",  # 样例 0.6b val-reg rec-0340
}

_TR_FULLWIDTH = str.maketrans("：（）；，", ":();,")
_WS = re.compile(r"\s+")


def normalize_disease(name: str) -> str:
    s = name.strip().translate(_TR_FULLWIDTH)
    s = _WS.sub("", s)
    s = s.rstrip(";；")
    return SYNONYMS.get(s, s)


def normalize_set(names: list[str]) -> set[str]:
    return {normalize_disease(n) for n in names if n and normalize_disease(n)}


def prf(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    """集合级 precision/recall/F1（None 安全：预测空集时 P=0）."""
    p, g = normalize_set(pred), normalize_set(gold)
    if not p:
        return 0.0, (1.0 if not g else 0.0), (1.0 if not g else 0.0)
    tp = len(p & g)
    prec = tp / len(p)
    rec = tp / len(g) if g else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def parse_diseases_tag(text: str) -> list[str] | None:
    """从模型输出解析 <diseases>…</diseases> 列表；解析失败/缺块返回 None（格式轴记败）."""
    m = re.search(r"<diseases>(.*?)</diseases>", text, re.S)
    if not m:
        return None
    items = [x for x in re.split(r"[;；]", m.group(1)) if x.strip() and x.strip() not in ("无", "数据异常", "缺少关键诊断信息")]
    return items
