#!/usr/bin/env python3
"""MED-E E1 长尾分层离线读数（零 GPU；D-09 §2 口径复核 + 更正 + 全配方分层）.

2026-09-30 E1 复核更正（对 D-09 的预写口径）：
  - test-ood 60 条 = 全病 corpus 单例记录池（build_dataset.py singleton_pool + R0a 断言
    train∩ood 疾病集 = ∅）→ 61 处 gold 引用 **100% train 零出现，构造性**；
    "62 处引用 29 处零出现"口径无法复现，作废。X1（unseen 缺口=知识层）结论被强化。
  - 频次分层的有效空间在 val-reg（seen 长尾）：gold 引用 train 频次 2~13 连续分布，
    桶 = 条目 gold 最低 train 频次 L(2-3)/M(4-5)/H(≥6)。
  - test-ood 侧读数改为"命名命中"：引用级 recall（=unseen 命名率）+ 命中病名清单。

疾病匹配与评测器同源（tools/normalize.py normalize_disease → 集合交），保证与轴①口径一致。
train 频次源 = swift-train-f1.jsonl assistant 目标疾病块（三臂同 N 同底料，f1 为代表）。

用法：python3 tools/med_e_stratify.py   # 产物落 output/med-e/stratify.json + stdout 表
     评分 JSON 缺省读 output/；核对公开版证据：EVIDENCE_ROOT=evidence python3 tools/med_e_stratify.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from normalize import normalize_disease, parse_diseases_tag  # noqa: E402

RAW = ROOT / "data/raw/medical-record-diagnosis.json"
TRAIN_F1 = ROOT / "data/med0/swift-train-f1.jsonl"
EVAL_OOD = ROOT / "data/med0/eval-test-ood.jsonl"
EVAL_VAL = ROOT / "data/med0/eval-val-reg.jsonl"
OUT_DIR = ROOT / "output/med-e"
EVIDENCE = Path(os.environ.get("EVIDENCE_ROOT", ROOT / "output"))  # 评分 JSON 根目录（公开版证据在 evidence/）

# 全配方列 score 坐标（波次目录, tag）——E1a 盘点表同源
CONFIGS = [
    ("零样本", "med-a/score-14b-val-reg.json", "med-a/score-14b-test-ood.json"),
    ("f1", "med-b/score-sft-14b-f1-val-reg.json", "med-b/score-sft-14b-f1-test-ood.json"),
    ("f2", "med-b/score-sft-14b-f2-val-reg.json", "med-b/score-sft-14b-f2-test-ood.json"),
    ("f3c", "med-b/score-sft-14b-f3c-val-reg.json", "med-b/score-sft-14b-f3c-test-ood.json"),
    ("cg", "med-c/score-sft-medc-cg-val-reg.json", "med-c/score-sft-medc-cg-test-ood.json"),
    ("cm", "med-c/score-sft-medc-cm-val-reg.json", "med-c/score-sft-medc-cm-test-ood.json"),
    ("cn", "med-c/score-sft-medc-cn-val-reg.json", "med-c/score-sft-medc-cn-test-ood.json"),
    ("cn-low", "med-c/score-sft-medc-cnlow-val-reg.json", "med-c/score-sft-medc-cnlow-test-ood.json"),
    ("dpo-s1", "med-d/score-dpo-s1-val-reg.json", "med-d/score-dpo-s1-test-ood.json"),
    ("dpo-s3", "med-d/score-dpo-s3-val-reg.json", "med-d/score-dpo-s3-test-ood.json"),
    ("dpo-s9", "med-d/score-dpo-s9-val-reg.json", "med-d/score-dpo-s9-test-ood.json"),
]

VAL_BUCKETS = [("L", 2, 3), ("M", 4, 5), ("H", 6, float("inf"))]  # 条目 gold 最低 train 频次


def train_freq() -> Counter:
    freq: Counter = Counter()
    for ln in TRAIN_F1.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        for d in parse_diseases_tag(row["messages"][-1]["content"]) or []:
            freq[normalize_disease(d)] += 1
    return freq


def corpus_caliber() -> dict:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    freq = Counter()
    for rec in raw:
        for d in rec["diseases"]:
            freq[normalize_disease(d)] += 1
    n_dis, n_sin = len(freq), sum(1 for v in freq.values() if v == 1)
    return {"records": len(raw), "diseases": n_dis, "singleton": n_sin,
            "singleton_pct": round(n_sin / n_dis * 100, 1)}


def bucket_of(minfreq: float) -> str:
    for name, lo, hi in VAL_BUCKETS:
        if lo <= minfreq <= hi:
            return name
    return "?"


def item_minfreq(item: dict, tfreq: Counter) -> int:
    return min(tfreq.get(normalize_disease(d), 0) for d in item["gold"]["diseases"])


def stratified(score_recs: dict, item_bucket: dict[str, str]) -> dict:
    per = {}
    for b, _, _ in VAL_BUCKETS:
        rs = [score_recs[i] for i, v in item_bucket.items() if v == b]
        f1s = [r["disease_f1"] for r in rs]
        covs = [r["coverage"] for r in rs if r.get("coverage") is not None]
        per[b] = {"n": len(rs),
                  "f1_macro": round(sum(f1s) / len(f1s), 3) if f1s else None,
                  "cov_macro": round(sum(covs) / len(covs), 3) if covs else None,
                  "fmt_ok": sum(1 for r in rs if r["fmt_ok"])}
    return per


def main() -> int:
    tfreq = train_freq()
    ood = [json.loads(ln) for ln in EVAL_OOD.read_text(encoding="utf-8").splitlines() if ln.strip()]
    val = [json.loads(ln) for ln in EVAL_VAL.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # ---- 口径复核（更正后的设计事实）----
    refs: Counter = Counter()  # norm 病名 -> train 频次（test-ood 去重）
    n_item_refs = sum(len(it["gold"]["diseases"]) for it in ood)
    for it in ood:
        for d in it["gold"]["diseases"]:
            refs[normalize_disease(d)] = tfreq.get(normalize_disease(d), 0)
    caliber = {
        "corpus": corpus_caliber(),
        "ood_refs": {"items": len(ood), "item_refs": n_item_refs,
                     "distinct": len(refs), "train_zero": sum(1 for v in refs.values() if v == 0),
                     "note": "全病 corpus 单例池 + R0a 断言 → 零出现是构造性 100%"},
        "val_refs_dist": dict(sorted(Counter(
            tfreq.get(normalize_disease(d), 0) for it in val for d in it["gold"]["diseases"]).items())),
    }

    # ---- 条目桶 ----
    val_bucket = {it["id"]: bucket_of(item_minfreq(it, tfreq)) for it in val}

    out = {"caliber": caliber, "val_buckets": dict(Counter(val_bucket.values())), "configs": {}}

    for name, val_rel, ood_rel in CONFIGS:
        s_val = json.loads((EVIDENCE / val_rel).read_text(encoding="utf-8"))
        s_ood = json.loads((EVIDENCE / ood_rel).read_text(encoding="utf-8"))
        recs_val = {r["id"]: r for r in s_val["records"]}
        recs_ood = {r["id"]: r for r in s_ood["records"]}

        # 聚合复核（防错位对齐）
        for s, recs in ((s_val, recs_val), (s_ood, recs_ood)):
            recalc = sum(r["disease_f1"] for r in recs.values()) / len(recs)
            assert abs(s["aggregate"]["disease_f1_macro"] - recalc) < 1e-9, f"{name}: 聚合对不上"

        # test-ood 引用级命中（unseen 命名率）+ 命中病名清单
        hit_names, hit = [], 0
        for it in ood:
            preds = {normalize_disease(p) for p in recs_ood[it["id"]]["pred_diseases"]}
            for d in it["gold"]["diseases"]:
                nd = normalize_disease(d)
                if nd in preds:
                    hit += 1
                    hit_names.append(d)
        ood_read = {
            "f1_macro": round(s_ood["aggregate"]["disease_f1_macro"], 3),
            "cov_macro": round(s_ood["aggregate"]["coverage_macro"], 3),
            "ref_recall": round(hit / n_item_refs, 3),
            "ref_hit": f"{hit}/{n_item_refs}",
            "hit_diseases": hit_names,
        }

        out["configs"][name] = {
            "val_strata": stratified(recs_val, val_bucket),
            "val_f1_macro": round(s_val["aggregate"]["disease_f1_macro"], 3),
            "ood": ood_read,
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stratify.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    c = caliber
    print(f"语料口径: {c['corpus']['records']} 条 / {c['corpus']['diseases']} 病种 / "
          f"singleton {c['corpus']['singleton']} = {c['corpus']['singleton_pct']}%")
    r = c["ood_refs"]
    print(f"ood 引用口径: {r['items']} 条 / {r['item_refs']} 处引用 / 去重 {r['distinct']} 病种 / "
          f"train 零出现 {r['train_zero']}（{r['note']}）")
    print(f"val-reg gold 引用 train 频次分布: {c['val_refs_dist']}")
    print(f"val-reg 条目桶 (L=2-3/M=4-5/H=≥6): {out['val_buckets']}")
    print()
    print("| 配方 | val F1 | L n/F1 | M n/F1 | H n/F1 | ood F1 | ood cov | 命名命中 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, _, _ in CONFIGS:
        cfg = out["configs"][name]
        v = cfg["val_strata"]
        cells = [f"{v[k]['n']}/{v[k]['f1_macro']}" for k, _, _ in VAL_BUCKETS]
        o = cfg["ood"]
        print(f"| {name} | {cfg['val_f1_macro']} | " + " | ".join(cells) +
              f" | {o['f1_macro']} | {o['cov_macro']} | {o['ref_hit']} |")
    print(f"\n落卷: {OUT_DIR / 'stratify.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
