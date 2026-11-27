#!/usr/bin/env python3
"""MED-0 数据基建：原始病历数据集 → 切分 + 训练/评测夹具（PLAN §4 MED-0）.

输入（data/raw/，指纹见 PROVENANCE.md）：
  - medical-record-diagnosis.json      1337 例原始记录（condition/diseases/analyses/diagnoses）
  - medical-record-diagnosis-sft.jsonl 1337 行官方 SFT 版（与原始版按下标 1:1 对齐，已验证）

产出（data/med0/）：
  - swift-train-f1.jsonl    F1 官方式目标（user/target 均逐字节复用官方行）
  - swift-train-f2.jsonl    F2 质量过滤目标（仅 score≥2 分析入分条依据）
  - swift-train-f3c.jsonl   F3-concise 目标（<diseases> 同官方 + 依据改用 diagnoses.reason 凝练分条；
                             diagnoses=None 的记录回退 F1 目标并打 fallback 标记——保三臂同 N，臂间可比）
  - eval-{val-reg,test-ood,test-rob}.jsonl   评测原料（user 逐字 + gold 全量，评测器后建于 MED-A）
  - split-manifest.json     切分清单（含去重指纹、分层配额、判决断言结果）

切分（疾病粒度防泄漏，seed=20260925，D-05）：
  1. 去重：condition 全文哈希，保首次出现（调研实测 22 组 23 条冗余）
  2. test-ood 60：全部疾病均为 corpus 单例的记录池（调研实测 381 例）按字段组合×性别分层抽样
     ——单例疾病的记录彼此疾病不相交，天然保证 train 看不到这些病
  3. val-reg 80 + test-rob 基材 20：全部疾病频次 ≥4 的记录池，贪心抽取且保 R0c
     （每个评测疾病在 train 剩 ≥2 例）
  4. train = 其余

预注册判决（MED-0，见 PLAN §4）：
  R0a 切分零泄漏：train 疾病集 ∩ test-ood 疾病集 = ∅（断言）
  R0b fidelity 自检：条数守恒 · 角色序 · user 逐字 · F1 目标重建与官方逐字节相等
  R0c 评测病不抽干：val-reg/test-rob 的每个疾病在 train ≥2 例（断言）

用法：
    python3 tools/build_dataset.py              # 构建 + 自检
    python3 tools/build_dataset.py --sample 3   # 追加打印抽检样本
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
OUT = REPO / "data" / "med0"
SEED = 20260925
SPLIT = {"test_ood": 60, "val_reg": 80, "rob_base": 20}
CORE_FIELDS = ("chiefComplaint", "historyOfPresentIllness", "physicalExamination", "auxiliaryExamination")
GARBLE_ALPHABET = "仩乛乱碼檟篿纊蠿衋蟻覑鬱齾"

# ---------------------------------------------------------------- 基础件


def cond_hash(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec["condition"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def combo_bucket(rec: dict) -> str:
    have = tuple(f for f in CORE_FIELDS if rec["condition"].get(f) not in (None, "", []))
    return "+".join(f[:4] for f in have) or "仅主诉外字段"


def diseases_of(rec: dict) -> list[str]:
    return list(rec["diseases"])


def render_diseases_block(diseases: list[str]) -> str:
    return "<diseases>\n" + ";".join(diseases) + ";\n</diseases>"


def numbered_items(descs: list[str]) -> str:
    return "\n".join(f"{i}. {d}" for i, d in enumerate(descs, 1))


def build_target(diseases: list[str], descs: list[str]) -> str:
    return render_diseases_block(diseases) + "\n" + numbered_items(descs)


def garble(user_msg: str, id_: str) -> str:
    """确定性乱码：非换行字符按 (下标 + id 哈希) 从固定字母表取字（X2/rob 条件·机械可导出版）."""
    offset = int(hashlib.sha256(id_.encode()).hexdigest()[:6], 16)
    out = []
    for i, ch in enumerate(user_msg):
        out.append(ch if ch == "\n" else GARBLE_ALPHABET[(i + offset) % len(GARBLE_ALPHABET)])
    return "".join(out)


def minimize(user_msg: str) -> str:
    """极简条件：只保留 性别/年龄/主诉 行（system prompt 规则 6/7 的触发形态）."""
    return "\n".join(ln for ln in user_msg.splitlines() if re.match(r"^(性别|年龄|主诉)：", ln))


# ---------------------------------------------------------------- 主流程


def load() -> tuple[list[dict], list[dict]]:
    raw = json.loads((RAW / "medical-record-diagnosis.json").read_text(encoding="utf-8"))
    sft = [json.loads(ln) for ln in (RAW / "medical-record-diagnosis-sft.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(raw) == len(sft) == 1337, f"源条数异常：{len(raw)}/{len(sft)}"
    # 对齐验证：sft user 行的行数与 condition 非空字段数一致是弱校验，直接用官方 zip 逐条验 analyses 计数
    for rec, row in zip(raw, sft):
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"], "官方行角色序异常"
        assert len(re.findall(r"^\d+\. ", row["messages"][-1]["content"], re.M)) == len(rec["analyses"]), \
            f"官方行与原始记录下标错位 @ analyses 计数"
    return raw, sft


def dedup(raw: list[dict], sft: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """按 condition 哈希去重（保首个）。重复组≠纯冗余：2026-09-28 复核定性为**同病历双标注**
    （病历同+疾病列表同+推理链独立成套，22 组无一例外；见 reports/med-0-data.md §6 补记），
    组语义随 manifest 落盘，供 MED-D 同输入偏好对 / 多参考评测 / 一致性探针取用。"""
    seen, kept_raw, kept_sft, members = {}, [], [], {}
    for raw_i, (rec, row) in enumerate(zip(raw, sft)):
        h = cond_hash(rec)
        if h in seen:
            members[h].append(raw_i)
        else:
            seen[h] = len(kept_raw)
            members[h] = [raw_i]
            kept_raw.append(rec)
            kept_sft.append(row)
    groups = []
    for h, raw_ix in members.items():
        if len(raw_ix) < 2:
            continue
        first = raw[raw_ix[0]]

        def all_same(key: str) -> bool:
            gold = json.dumps(first[key], sort_keys=True, ensure_ascii=False)
            return all(json.dumps(raw[i][key], sort_keys=True, ensure_ascii=False) == gold for i in raw_ix)

        same_diseases = all_same("diseases")
        groups.append({
            "hash": h, "kept_id": f"rec-{seen[h]:04d}",
            "kept_raw_index": raw_ix[0], "dup_raw_indices": raw_ix[1:],
            "same_diseases": same_diseases, "same_analyses": all_same("analyses"),
            "same_diagnoses": all_same("diagnoses"),
            "kind": "double_annotated" if same_diseases else "label_conflict",
        })
    return kept_raw, kept_sft, groups


def stratified_pick(pool: list[int], combo_of: dict, gender_of: dict, n: int, rng: random.Random) -> list[int]:
    """按 (字段组合 × 性别) 双层配额的确定性抽样."""
    groups: dict[tuple[str, str], list[int]] = {}
    for i in pool:
        groups.setdefault((combo_of[i], gender_of[i] or "未知"), []).append(i)
    for g in groups.values():
        rng.shuffle(g)
    quota = {k: round(n * len(v) / len(pool)) for k, v in groups.items()}
    picked = [i for k in sorted(groups) for i in groups[k][: quota[k]]]
    # 配额取整可能溢出/不足，截齐或从剩余补
    if len(picked) > n:
        picked = picked[:n]
    elif len(picked) < n:
        rest = [i for i in pool if i not in set(picked)]
        rng.shuffle(rest)
        picked += rest[: n - len(picked)]
    return picked


def make_split(records: list[dict]) -> dict[str, list[int]]:
    rng = random.Random(SEED)
    freq = Counter(d for r in records for d in diseases_of(r))
    combo_of = {i: combo_bucket(r) for i, r in enumerate(records)}
    gender_of = {i: r["condition"].get("gender") or "" for i, r in enumerate(records)}

    singleton_pool = [i for i, r in enumerate(records) if all(freq[d] == 1 for d in diseases_of(r))]
    frequent_pool = [i for i, r in enumerate(records) if all(freq[d] >= 4 for d in diseases_of(r))]

    test_ood = set(stratified_pick(singleton_pool, combo_of, gender_of, SPLIT["test_ood"], rng))

    # val_reg + rob_base：贪心抽，保 R0c（每病在 train 剩 ≥2）
    candidates = [i for i in frequent_pool if i not in test_ood]
    rng.shuffle(candidates)
    val_reg, rob_base = [], []
    train_count = Counter(freq)
    for i in candidates:
        if len(val_reg) < SPLIT["val_reg"] and all(train_count[d] > 2 for d in diseases_of(records[i])):
            val_reg.append(i)
            for d in diseases_of(records[i]):
                train_count[d] -= 1
        elif len(rob_base) < SPLIT["rob_base"] and all(train_count[d] > 2 for d in diseases_of(records[i])):
            rob_base.append(i)
            for d in diseases_of(records[i]):
                train_count[d] -= 1
        if len(val_reg) >= SPLIT["val_reg"] and len(rob_base) >= SPLIT["rob_base"]:
            break
    train = [i for i in range(len(records)) if i not in set(val_reg) | set(rob_base) | test_ood]
    return {"train": train, "val_reg": val_reg, "test_ood": sorted(test_ood), "rob_base": rob_base}


def build_forms(records: list[dict], sft_rows: list[dict], split: dict) -> dict[str, list[dict]]:
    system_prompt = sft_rows[0]["messages"][0]["content"]
    assert all(r["messages"][0]["content"] == system_prompt for r in sft_rows), "官方 system prompt 非全局一致"

    out = {"f1": [], "f2": [], "f3c": []}
    meta = {"f2_fallback": 0, "f3c_fallback": 0}
    for i in split["train"]:
        rec, row = records[i], sft_rows[i]
        user = row["messages"][1]["content"]
        official_target = row["messages"][-1]["content"]
        rid = f"rec-{i:04d}"

        # F1：逐字节复用官方
        out["f1"].append({"messages": [{"role": "system", "content": system_prompt},
                                       {"role": "user", "content": user},
                                       {"role": "assistant", "content": official_target}],
                          "id": rid, "form": "f1"})

        # F2：仅 score≥2 分析
        keep = [a["description"] for a in rec["analyses"] if a["score"] >= 2]
        if keep:
            tgt = build_target(diseases_of(rec), keep)
        else:  # 全低分记录：回退官方目标（保三臂同 N）
            tgt, meta["f2_fallback"] = official_target, meta["f2_fallback"] + 1
        out["f2"].append({"messages": [{"role": "system", "content": system_prompt},
                                       {"role": "user", "content": user},
                                       {"role": "assistant", "content": tgt}],
                          "id": rid, "form": "f2"})

        # F3-concise：依据用 diagnoses.reason
        if rec["diagnoses"]:
            tgt = build_target(diseases_of(rec), [d["reason"] for d in rec["diagnoses"]])
        else:
            tgt, meta["f3c_fallback"] = official_target, meta["f3c_fallback"] + 1
        out["f3c"].append({"messages": [{"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user},
                                        {"role": "assistant", "content": tgt}],
                           "id": rid, "form": "f3c"})
    return out, meta


def build_evals(records: list[dict], sft_rows: list[dict], split: dict) -> dict[str, list[dict]]:
    def gold(i: int) -> dict:
        rec = records[i]
        return {"diseases": diseases_of(rec),
                "conclusions": [d["conclusion"] for d in (rec["diagnoses"] or [])],
                "reasons": [d["reason"] for d in (rec["diagnoses"] or [])],
                "analyses": rec["analyses"]}

    def base_row(i: int, extra: dict) -> dict:
        return {"id": f"rec-{i:04d}", "user": sft_rows[i]["messages"][1]["content"], "gold": gold(i), **extra}

    val = [base_row(i, {"split": "val_reg"}) for i in split["val_reg"]]
    ood = [base_row(i, {"split": "test_ood"}) for i in split["test_ood"]]
    rob = []
    for i in split["rob_base"]:
        rid = f"rec-{i:04d}"
        u = sft_rows[i]["messages"][1]["content"]
        rob.append(base_row(i, {"split": "test_rob", "condition_kind": "garbled", "expected_category": "数据异常",
                                "user": garble(u, rid)}))
        rob.append(base_row(i, {"split": "test_rob", "condition_kind": "minimal", "expected_category": "推测或缺少关键诊断信息",
                                "user": minimize(u)}))
    return {"eval-val-reg": val, "eval-test-ood": ood, "eval-test-rob": rob}


# ---------------------------------------------------------------- 自检（R0a/R0b/R0c）


def selfcheck(records, sft_rows, split, forms, evals, dup_groups) -> dict:
    n = len(records)
    train_ds = set(d for i in split["train"] for d in diseases_of(records[i]))
    ood_ds = set(d for i in split["test_ood"] for d in diseases_of(records[i]))
    eval_ds = set(d for i in split["val_reg"] + split["rob_base"] for d in diseases_of(records[i]))
    train_counter = Counter(d for i in split["train"] for d in diseases_of(records[i]))

    # R0a 零泄漏 + 评测病 seen
    assert not (train_ds & ood_ds), "R0a 失败：train 与 test_ood 共享疾病"
    assert eval_ds <= train_ds, "R0a' 失败：val_reg/test_rob 存在 train 未见疾病"
    # R0c 评测病不抽干
    starve = [d for d in eval_ds if train_counter[d] < 2]
    assert not starve, f"R0c 失败：评测疾病在 train <2 例：{starve[:5]}"
    # 切分守恒 + 不相交
    assert len(set(split["train"]) | set(split["val_reg"]) | set(split["rob_base"]) | set(split["test_ood"])) == n
    assert not (set(split["train"]) & set(split["test_ood"]))

    # R0b fidelity：F1 行与官方逐字节；重建式一致性；角色序
    f1_by_id = {r["id"]: r for r in forms["f1"]}
    for i in split["train"]:
        rid = f"rec-{i:04d}"
        row = sft_rows[i]
        m = f1_by_id[rid]["messages"]
        assert [x["role"] for x in m] == ["system", "user", "assistant"], f"{rid} 角色序"
        assert m[1]["content"] == row["messages"][1]["content"], f"{rid} user 不逐字"
        assert m[2]["content"] == row["messages"][-1]["content"], f"{rid} F1 目标不逐字"
        # 重建一致性（F2/F3c 的构造器在 F1 上重建必须逐字节，否则形态间不可比）
        rec = records[i]
        rebuilt = build_target(diseases_of(rec), [a["description"] for a in rec["analyses"]])
        assert rebuilt == row["messages"][-1]["content"], f"{rid} 目标重建 ≠ 官方（构造器漂移）"
    for form in ("f2", "f3c"):
        for r in forms[form]:
            assert len(r["messages"]) == 3 and r["messages"][1]["content"] == f1_by_id[r["id"]]["messages"][1]["content"]
            assert "<diseases>" in r["messages"][-1]["content"] and "1. " in r["messages"][-1]["content"]

    return {
        "n_source": 1337, "n_dedup": n,
        "n_dup_groups": len(dup_groups),
        "n_removed_dups": sum(len(g["dup_raw_indices"]) for g in dup_groups),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "unique_diseases_total": len(train_ds | ood_ds | eval_ds),
        "ood_singleton_pool": 381,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    raw, sft = load()
    records, sft_rows, dup_groups = dedup(raw, sft)
    split = make_split(records)
    forms, meta = build_forms(records, sft_rows, split)
    evals = build_evals(records, sft_rows, split)
    stats = selfcheck(records, sft_rows, split, forms, evals, dup_groups)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [("swift-train-f1.jsonl", forms["f1"]), ("swift-train-f2.jsonl", forms["f2"]),
                       ("swift-train-f3c.jsonl", forms["f3c"]), ("eval-val-reg.jsonl", evals["eval-val-reg"]),
                       ("eval-test-ood.jsonl", evals["eval-test-ood"]), ("eval-test-rob.jsonl", evals["eval-test-rob"])]:
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    manifest = {
        "seed": SEED, "split_rule": "疾病粒度（singleton→ood / freq≥4→评测 / 其余 train）",
        "splits": {k: [f"rec-{i:04d}" for i in v] for k, v in split.items()},
        "removed_duplicates_superseded_by": "duplicate_groups",
        "duplicate_groups": dup_groups,
        "fallbacks": meta, "checks": stats,
        "source_sha256": "见 data/raw/SHA256SUMS",
    }
    (OUT / "split-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print("fallbacks:", meta)
    print("R0a/R0b/R0c 断言全绿 ✓")

    if args.sample:
        for r in forms["f2"][: args.sample] + forms["f3c"][: args.sample]:
            print("=" * 60, r["id"], r["form"])
            print(r["messages"][-1]["content"][:300])


if __name__ == "__main__":
    main()
