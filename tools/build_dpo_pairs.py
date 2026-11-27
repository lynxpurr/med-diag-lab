#!/usr/bin/env python3
"""MED-D D1：300 对 DPO 偏好数据构建 → data/medd/dpo-pairs.jsonl.

两来源（X4 工作表判定口径，research/2026-09-28-x4-pairs-worksheet.md）：
  A 源（294 对）：train split 内 record，同病历 chosen = score=3 项集 vs rejected = score≤2 项集
    ——X4 已判 score 信号可信（Y 96.7% ≥ 80% 线），直入不加人核层
  B 源（6 对）：双标注对实质差异 K3+D3 —— K（rec-0154/0427/0823）chosen=kept 链，
    D（rec-0740/0793/0923）chosen=dup 链（X4 §B 逐条人核代行判定）

组装：沿用 build_dataset.build_target 形态（<diseases> 英文分号块 + 数字序号依据链），
两侧同 diseases 同 user——偏好信号只在依据链质量（PLAN §4："DPO 教的是结构完整性"）。
泄题防线：pair 仅取自 train split（1154），val-reg/test-ood/rob_base 一字不入（断言）。
抽样：random.Random(20260925).sample(sorted(资格池), 294)——与 X4 的 30 对同法同 seed，
前 30 对即 X4 已人核样本（复跑确定性）。

长度资格（2026-09-29 D2 显存适配追加，校准块性质）：A 源资格 ∧ max(侧长) ≤ LEN_CAP tok。
背景：14B QLoRA DPO 的 micro-batch = chosen+rejected 拼批 [2, L]，bnb 反向去量化 × 边界激活
在长对上 OOM（s1 三次实录 + 截断探针实测 L1613→12.07GB / L1400→11.50GB allocated，seq 伸缩
~2.7MB/token）；共享卡常驻（常驻服务 2.76GB + 第三方进程 608MiB）下预算 ~12.05GB。
cap=1350 → micro-batch ≤2700 tok，探针背书余量 ~0.6GB。侧长来自 tools/profile_dpo_lengths.py
离线实测（data/medd/length-profile.json，1043 资格全量）；B 源 6 对整链实测 1142-1317 全保留。
超限对由同池等量替换（保 300 对预算与步数剂量不动），组成变化入 manifest 记账。
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
MED0 = REPO / "data" / "med0"
MEDD = REPO / "data" / "medd"

SEED = 20260925
N_A, N_B = 294, 6
LEN_CAP = 1350  # DPO 显存适配长度资格（见模块 docstring）
# B 源 6 对整链 max 侧长（2026-09-29 swift encode 实测，kept 链 vs dup 链）
B_LEN = {"rec-0154": 1272, "rec-0427": 1219, "rec-0823": 1255,
         "rec-0740": 1317, "rec-0793": 1169, "rec-0923": 1142}
# X4 §B 表非 T 行（K=kept 优 → chosen=kept；D=dup 优 → chosen=dup）
B_SOURCE = {"rec-0154": "K", "rec-0427": "K", "rec-0823": "K",
            "rec-0740": "D", "rec-0793": "D", "rec-0923": "D"}

rid2idx = lambda r: int(r.split("-")[1])


def build_target(diseases: list[str], descs: list[str]) -> str:
    """与 tools/build_dataset.py 逐字节同构（diseases 块 + 编号链）。"""
    return "<diseases>\n" + ";".join(diseases) + ";\n</diseases>" + \
        "\n" + "\n".join(f"{i}. {d}" for i, d in enumerate(descs, 1))


def main() -> int:
    raw = json.loads((RAW / "medical-record-diagnosis.json").read_text(encoding="utf-8"))
    sft = [json.loads(ln) for ln in (RAW / "medical-record-diagnosis-sft.jsonl")
           .read_text(encoding="utf-8").splitlines() if ln.strip()]
    manifest = json.loads((MED0 / "split-manifest.json").read_text(encoding="utf-8"))
    assert len(raw) == len(sft) == 1337
    system_prompt = sft[0]["messages"][0]["content"]
    assert all(r["messages"][0]["content"] == system_prompt for r in sft), "system 非全局一致"

    train_ids = set(manifest["splits"]["train"])
    eval_ids = set(manifest["splits"]["val_reg"]) | set(manifest["splits"]["test_ood"]) \
        | set(manifest["splits"]["rob_base"])

    # ---- A 源资格池：train ∧ score=3 链 ∧ score≤2 链并存（X4 同口径，应 1043）----
    elig = sorted(i for i in map(rid2idx, train_ids)
                  if 3.0 in [a["score"] for a in raw[i]["analyses"]]
                  and any(a["score"] <= 2 for a in raw[i]["analyses"]))
    assert len(elig) == 1043, f"A 源资格池异常：{len(elig)}（X4 实测 1043）"
    b_indices = {rid2idx(r) for r in B_SOURCE}
    profile = json.loads((MEDD / "length-profile.json").read_text(encoding="utf-8"))
    assert len(profile) == 1043, "length-profile.json 与资格池不齐（先跑 tools/profile_dpo_lengths.py）"
    for rid, side in B_LEN.items():
        assert side <= LEN_CAP, f"{rid} B 对整链超长（{side}）——重测并调整"
    pool = [i for i in elig if i not in b_indices
            and max(profile[f"rec-{i:04d}"]["len_chosen"], profile[f"rec-{i:04d}"]["len_rejected"]) <= LEN_CAP]
    uncapped_n = len([i for i in elig if i not in b_indices])
    rng = random.Random(SEED)
    a_pick = rng.sample(pool, N_A)

    rows, audit = [], {"a": [], "b": []}

    # ---- A 源对 ----
    for i in a_pick:
        rec, row = raw[i], sft[i]
        chosen_descs = [a["description"] for a in rec["analyses"] if a["score"] == 3]
        rejected_descs = [a["description"] for a in rec["analyses"] if a["score"] <= 2]
        assert chosen_descs and rejected_descs
        rows.append({
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": row["messages"][1]["content"]},
                         {"role": "assistant", "content": build_target(rec["diseases"], chosen_descs)}],
            "rejected_response": build_target(rec["diseases"], rejected_descs),
            "id": f"rec-{i:04d}", "source": "a-score",
        })
        audit["a"].append({"id": f"rec-{i:04d}", "n_chosen": len(chosen_descs),
                           "n_rejected": len(rejected_descs)})

    # ---- B 源对（双标注，同 user 同 diseases，链整套互换）----
    gmap = {g["kept_id"]: g for g in manifest["duplicate_groups"]}
    for rid, kd in B_SOURCE.items():
        g = gmap[rid]
        ki, di = g["kept_raw_index"], g["dup_raw_indices"][0]
        assert rid in train_ids and len(g["dup_raw_indices"]) == 1
        kept_rec, dup_rec = raw[ki], raw[di]
        assert kept_rec["diseases"] == dup_rec["diseases"], f"{rid} 双标注 diseases 不同"
        assert sft[ki]["messages"][1]["content"] == sft[di]["messages"][1]["content"], \
            f"{rid} 双标注 user 不同（condition 应同）"
        kept_chain = [a["description"] for a in kept_rec["analyses"]]
        dup_chain = [a["description"] for a in dup_rec["analyses"]]
        chosen_chain, rejected_chain = (kept_chain, dup_chain) if kd == "K" else (dup_chain, kept_chain)
        assert chosen_chain != rejected_chain, f"{rid} 两链相同（X4 判 K/D 应实质差异）"
        rows.append({
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": sft[ki]["messages"][1]["content"]},
                         {"role": "assistant", "content": build_target(kept_rec["diseases"], chosen_chain)}],
            "rejected_response": build_target(kept_rec["diseases"], rejected_chain),
            "id": rid, "source": "b-dup", "chosen_side": "kept" if kd == "K" else "dup",
        })
        audit["b"].append({"id": rid, "verdict": kd, "n_chosen": len(chosen_chain),
                           "n_rejected": len(rejected_chain)})

    # ---- 断言门 ----
    assert len(rows) == 300 and len(audit["a"]) == N_A and len(audit["b"]) == N_B
    all_ids = [r["id"] for r in rows]
    assert len(set(all_ids)) == 300, "record 重复入对"
    assert not (set(all_ids) & eval_ids), "泄题：pair 越界入评测 split"
    assert all(r["messages"][2]["content"] != r["rejected_response"] for r in rows), "退化对"

    # ---- 落盘 ----
    MEDD.mkdir(parents=True, exist_ok=True)
    pairs_path = MEDD / "dpo-pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                          encoding="utf-8")
    sha = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    pair_manifest = {
        "wave": "med-d", "built": "2026-09-29", "seed": SEED, "n_pairs": 300,
        "sources": {"a-score": N_A, "b-dup": N_B},
        "length_cap": {"tok": LEN_CAP, "rationale": "DPO 显存适配（s1 三次 OOM + 截断探针 L1400→11.50GB "
                      "allocated 实测背书，micro-batch≤2700 tok）；A 池保留率与替换数见 a_pool",
                       "b_pairs_max_side": B_LEN},
        "a_pool": {"eligible_in_train": 1043, "uncapped_minus_b": uncapped_n, "capped_minus_b": len(pool),
                   "sampled": N_A,
                   "method": f"random.Random({SEED}).sample(sorted(capped_pool), {N_A})（X4 同法）"},
        "b_source": B_SOURCE,
        "format": "messages(system+user+assistant=chosen裸) + rejected_response(裸)；"
                  "组装形态=build_target 同构（diseases 英文分号块+编号链）",
        "leak_guard": "pair 全部 ∈ train split；val-reg/test-ood/rob_base 断言零交集",
        "audit": audit,
        "sha256_dpo_pairs": sha,
    }
    # x4_overlap 简化重算（上行 walrus 写法可读性差，直接重抽一次）
    x4_30 = set(random.Random(SEED).sample(elig, 30))
    pair_manifest["a_pool"]["x4_overlap"] = len(set(a_pick) & x4_30)
    (MEDD / "manifest.json").write_text(json.dumps(pair_manifest, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    nch = [x["n_chosen"] for x in audit["a"]]
    nrj = [x["n_rejected"] for x in audit["a"]]
    print(f"[build] 300 对落 {pairs_path}")
    print(f"[build] sha256={sha[:16]}…")
    print(f"[build] A 源 chosen 项数 min/med/max = {min(nch)}/{sorted(nch)[len(nch)//2]}/{max(nch)}；"
          f"rejected 项数 min/med/max = {min(nrj)}/{sorted(nrj)[len(nrj)//2]}/{max(nrj)}")
    print(f"[build] A 源与前 30 对 X4 人核样本重合 {pair_manifest['a_pool']['x4_overlap']} 条（同 seed 前 30）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
