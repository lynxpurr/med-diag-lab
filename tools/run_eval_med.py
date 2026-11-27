#!/usr/bin/env python3
"""MED-A 零训练基线评测器（五轴判分 v1，D-04 全确定性零 LLM judge）.

流程：prep 渲染 → swift infer 子进程 → 保序对齐判分 → 证据只增不改。

五轴（口径 v1.1；开发期的口径修正已由 gold 自校准回路定案，见下方【修正】）：
  ① 疾病命中  <diseases> 解析（normalize.py）→ 集合 P/R/F1，宏平均；仅 val-reg/test-ood
              （test-rob 的 gold 是基材原图疾病，对乱码/极简输入无判分意义 → N/A）
  ② 格式合规  块可解析 ∧ 块内无中文分号 ∧ 依据分条序号正则。
              【修正 1】"目标内无中文分号"收窄为"块内"——gold 依据散文含合法中文分号
              （val-reg×1 / test-rob×2），system prompt 规则 2 只约束疾病列表；
              散文中文分号另记 info 不判败。
  ③ 依据覆盖  gold.reasons 抽证据原子（数值+检验/体征单位 token ∋ 检查/体征关键词），
              生成依据命中率（宏平均，原子数 0 的条目不计）；仅 val-reg/test-ood。
              【修正 2】原子单位词表限检验/体征值（%|℃|mmHg|拉丁单位|次/分…），年龄/时长
              类不计——gold 标注含标注者补充的产史/时长数值（rec-0751 等 3 条 5 原子），
              输入不可追溯；且安全意图（X5）本就是防编造检验值。
              【修正 3 · v1.1】单位表再剔除 个 与裸 次（轴③人核校准：rec-1303 "1个月"
              →"1个"、rec-1034 "5～6次"→"6次" 假原子——症状史计数非检验值；次/分保留）。
  ④ 鲁棒命中  仅 test-rob：garbled → 块含"数据异常"；minimal → 疾病非空 或 块含
              "缺少关键诊断信息"；两者均要求依据非空。
  ⑥ 无病命中  仅 neg-draft（轴⑥探针集，MED-C 拍板材料，2026-09-28 扩）：解析列表恰为
              ["无"]（块含"无"∧无疾病名项）∧ 依据非空；"解释健康原因"无机械判据，
              软读数 neg_reason_ok（依据含健康类关键词）只记录不判败。
  ⑤ 幻觉计数  生成依据中"数值+检验/体征单位"原子不存在于输入病历的数量（归一化子串；
              年龄感知：token=N岁 且输入 年龄：N 算在场）；仅 val-reg/test-ood。

gold 自校准铁律：gold 目标 = build_target(gold.diseases, gold.reasons)
（build_dataset.py 构造器同源，单一事实源；test-rob 的 garbled 渲染为 数据异常 块）。
期望：val-reg/test-ood ①=1.0 ②=100% ③=1.0 ⑤=0；test-rob ②=100% ④=100%。
gold 不满分 = 评测器有 bug，禁止跑模型。

用法（python3 为系统解释器，评测器纯 CPU；infer 子命令内部调 ms-swift 的 swift，路径取
环境变量 SWIFT_BIN，缺省用 PATH 上的 swift）：
    python3 tools/run_eval_med.py gold  [eval集…]        # A1 自校准（门）
    python3 tools/run_eval_med.py prep  [eval集…]        # 渲 swift 推理输入（模型无关）
    python3 tools/run_eval_med.py infer <tag> <模型路径> <eval集> [--quant_bits 4] [--limit N]
    python3 tools/run_eval_med.py score <tag> <eval集>    # 判分 + 汇总落 output/med-a/
    python3 tools/run_eval_med.py table                   # 聚合 score-*.json → markdown 主表

ckpt 评测（adapters 模式）：模板/量化/思考开关从 ckpt 同级
args.json 自动恢复，不传 --model/--template/--quant_bits；<模型路径> 省略改传 --adapters；
默认 batch 1 + expandable_segments（14B 4bit 贴边先例，MED-A 实测）；--out 切产物波次目录
（prep 输入模型无关，缺省回读 output/med-a/ 的既有 infer-input，不重复渲染）：
    python3 tools/run_eval_med.py infer <tag> <eval集> --adapters <ckpt> --out output/med-b
    python3 tools/run_eval_med.py infer <tag> <eval集> --adapters <ckpt> --quant bf16 --out output/med-e
                                                        # ^ bf16 消口径列（MED-E）：load_args false 跳过
                                                        #   ckpt 量化恢复，基座/模板显式传，无量化加载
    python3 tools/run_eval_med.py score <tag> <eval集> --out output/med-b
    python3 tools/run_eval_med.py table --out output/med-b
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from build_dataset import build_target  # noqa: E402
from normalize import parse_diseases_tag, prf  # noqa: E402

MED0 = REPO / "data" / "med0"
RAW_SFT = REPO / "data" / "raw" / "medical-record-diagnosis-sft.jsonl"
SWIFT = Path(os.environ.get("SWIFT_BIN", "swift"))  # ms-swift venv 的 swift；缺省取 PATH 上的 swift（D-03）
OUT = REPO / "output" / "med-a"
# bf16 消口径列的显存/内存预算（键 0 = 0 号 GPU）。缺省值对应实验机：与常驻服务共享的 16GB 显卡 + 40GB 内存；
# 换机器用环境变量按显存与内存设置，例如 MED_BF16_MAX_MEMORY='{0: "20GiB", "cpu": "64GiB"}'
BF16_MAX_MEMORY = os.environ.get("MED_BF16_MAX_MEMORY", '{0: "9GiB", "cpu": "40GiB"}')
SETS = ("val-reg", "test-ood", "test-rob", "neg-draft", "negn-draft", "neg", "negn", "negt")
# 轴⑥无病评测集：2026-09-29 素材正式化（MED-C C1）——research/ 草稿 git mv 入 data/medc/，
# 正式键 neg/negn 指新位；neg-draft/negn-draft 保留为别名（MED-B 期证据文件名沿用，二键同源不改判）。
# negt = MED-C 无病臂训练素材的造数质量门探针集（data/medc/aug-neg.jsonl 抽 messages 列渲染）
EXTRA_SETS = {"neg": REPO / "data" / "medc" / "eval-neg.jsonl",       # neg-001~020 人工形态 A/B/C
               "negn": REPO / "data" / "medc" / "eval-negn.jsonl",     # negn-001~010 阴性化对照
               "negt": REPO / "data" / "medc" / "probe-negt.jsonl",     # negt-* 训练素材探针（不入判卷）
               "neg-draft": REPO / "data" / "medc" / "eval-neg.jsonl",
               "negn-draft": REPO / "data" / "medc" / "eval-negn.jsonl"}

# ---------------------------------------------------------------- 轴③⑤ 证据原子

# 数值+检验/体征单位：拉丁单位（mmHg/mmol/L/U/L/cm/mg…含 ×10ⁿ 科学计数）+ 显式符号 + 有限 CJK。
# 年龄/时长单位（岁年月周天日时小时分钟分）刻意不在表（见模块 docstring 修正 2）；
# 2026-09-28 v1.1 再剔除 个 与裸 次（人核校准发现"1个月"→"1个"、"5～6次"→"6次"假原子：
# 计数/频次是症状史复述不是检验值；心率类 次/分 保留）。
_LATIN = r"[A-Za-zμµ]+(?:\s*/\s*[A-Za-zμµ]+)*"
_CJK_U = "次/分|厘米|毫米|枚|片|粒|袋|支|万|度"
ATOM_RE = re.compile(
    r"\d+(?:\.\d+)?(?:\s*[×x]\s*10\s*[⁰¹²³⁴⁵⁶⁷⁸⁹]+\s*)?"
    r"(?:\s*(?:%|℃|°C|mmHg|" + _LATIN + "|" + _CJK_U + r"))"
)
_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}

# 检查/体征关键词（v1 起点表；错误分析期按 A4 人核追加，改动记报告）
KEYWORDS = (
    "压痛", "反跳痛", "肌紧张", "包块", "肿大", "水肿", "浮肿", "黄疸", "发绀", "苍白",
    "杂音", "啰音", "瞳孔", "腱反射", "肌张力", "淋巴结", "潮红", "皮疹", "出血点", "静脉曲张",
    "血常规", "尿常规", "便常规", "X线", "CT", "MRI", "超声", "心电图", "造影", "血沉", "生化",
    "WBC", "RBC", "Hb", "HGB", "PLT", "TSH", "AFP", "CEA",
)


def _norm(s: str) -> str:
    for k, v in _SUP.items():
        s = s.replace(k, v)
    s = s.replace("×", "x").replace("^", "")
    return re.sub(r"[\s\u3000]+", "", s).lower()


def value_atoms(text: str) -> list[str]:
    return [_norm(m.group(0)) for m in ATOM_RE.finditer(text)]


def keyword_atoms(text: str) -> list[str]:
    return [k for k in KEYWORDS if k in text]


def atom_present(tok: str, user_raw: str, user_norm: str) -> bool:
    if tok in user_norm:
        return True
    m = re.fullmatch(r"(\d+)岁", tok)
    return bool(m and re.search(rf"年龄[：:]\s*{m.group(1)}(?:\D|$)", user_raw))


# ---------------------------------------------------------------- 单条判分

def split_target(text: str) -> tuple[str | None, str]:
    """→ (diseases 块原文或 None, 依据尾部)."""
    m = re.search(r"<diseases>(.*?)</diseases>", text, re.S)
    if not m:
        return None, text
    return m.group(1), text[m.end():]


def numbered_ok(tail: str) -> bool:
    return bool(re.search(r"(?m)^\s*\d+[\.、]", tail) or re.search(r"\s\d+[\.、]\s", tail))


def score_item(item: dict, resp: str) -> dict:
    """五轴单条判分。轴适用面：rob 集 ①③⑤ 为 None（口径见模块 docstring）；neg 集
    （轴⑥无病探针）同 rob 处理——只有 ②格式 + ⑥无病命中有判分意义."""
    kind = item.get("condition_kind")
    is_rob = item.get("split") == "test_rob" and kind != "negative"
    is_neg = kind == "negative"
    block, tail = split_target(resp)
    parseable = block is not None
    block_cn_semi = parseable and "；" in block
    prose_cn_semi = "；" in tail
    fmt_ok = parseable and not block_cn_semi and numbered_ok(tail)

    pred = parse_diseases_tag(resp) if parseable else None
    reasons_txt = " ".join(item["gold"]["reasons"])
    tail_norm = _norm(tail)

    rec: dict = {
        "id": item["id"],
        "fmt_ok": fmt_ok,
        "fmt_detail": {"parseable": parseable, "block_cn_semicolon": block_cn_semi,
                       "numbered": numbered_ok(tail), "prose_cn_semicolon": prose_cn_semi},
        "reasons_nonempty": bool(tail.strip()),
    }
    if is_neg:  # 轴⑥无病命中（MED-C 拍板材料口径）：块恰为"无" ∧ 无疾病名项 ∧ 依据非空。
        # 不复用 parse_diseases_tag——它按轴①口径把"无"当哨兵词过滤（normalize.py L55），
        # 此处需要的是"块内容恰为无"本身：与轴①解析同源的分号切分，但不过滤哨兵词。
        items_raw = [x.strip() for x in re.split(r"[;；]", block or "") if x.strip()]
        neg_hit = parseable and items_raw == ["无"] and bool(tail.strip())
        neg_reason_ok = bool(re.search(r"正常|健康|参考区间|无异常|未见异常", tail))
        rec.update({"condition_kind": "negative", "neg_hit": neg_hit,
                    "neg_reason_ok": neg_reason_ok})
    elif not is_rob:
        p, r, f1 = prf(pred or [], item["gold"]["diseases"])
        atoms = value_atoms(reasons_txt) + keyword_atoms(reasons_txt)
        hits = [a for a in atoms if _norm(a) in tail_norm] if atoms else []
        u_raw, u_norm = item["user"], _norm(item["user"])
        halluc = [a for a in value_atoms(tail) if not atom_present(a, u_raw, u_norm)]
        rec.update({
            "disease_p": p, "disease_r": r, "disease_f1": f1,
            "pred_diseases": [x.strip() for x in (pred or [])],
            "coverage": (len(hits) / len(atoms)) if atoms else None,
            "n_atoms": len(atoms),
            "halluc_count": len(halluc), "halluc_atoms": halluc[:8],
        })
    else:
        kind = item["condition_kind"]
        if kind == "garbled":
            rob_hit = parseable and "数据异常" in block and bool(tail.strip())
        else:  # minimal：疾病非空（解析列表非空）或 块含缺少关键诊断信息
            rob_hit = parseable and bool(tail.strip()) and (
                bool(pred) or "缺少关键诊断信息" in block)
        rec.update({"condition_kind": kind, "rob_hit": rob_hit})
    return rec


# ---------------------------------------------------------------- gold 渲染与自校准

def load_eval(set_name: str) -> list[dict]:
    path = EXTRA_SETS.get(set_name) or MED0 / f"eval-{set_name}.jsonl"
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def system_prompt() -> str:
    """每次从官方 SFT 首行提取（与训练同串，单一事实源，不复制粘贴常量）."""
    first = json.loads(RAW_SFT.read_text(encoding="utf-8").splitlines()[0])
    return first["messages"][0]["content"]


def gold_target(item: dict) -> str:
    """gold 目标渲染（与训练构造器同源）：rob 的 garbled 条按 expected_category 渲 数据异常块."""
    if item.get("condition_kind") == "garbled":
        return build_target(["数据异常"], item["gold"]["reasons"])
    return build_target(item["gold"]["diseases"], item["gold"]["reasons"])


def aggregate(recs: list[dict], set_name: str) -> dict:
    agg: dict = {"n": len(recs),
                 "fmt_ok": sum(r["fmt_ok"] for r in recs)}
    if set_name.startswith("neg"):
        agg.update({"neg_hit": sum(r["neg_hit"] for r in recs),
                    "neg_reason_ok": sum(r["neg_reason_ok"] for r in recs)})
    elif set_name == "test-rob":
        for kind in ("garbled", "minimal"):
            rows = [r for r in recs if r["condition_kind"] == kind]
            agg[f"rob_hit_{kind}"] = sum(r["rob_hit"] for r in rows)
            agg[f"n_{kind}"] = len(rows)
    else:
        f1s = [r["disease_f1"] for r in recs]
        covs = [r["coverage"] for r in recs if r["coverage"] is not None]
        agg.update({
            "disease_f1_macro": sum(f1s) / len(f1s),
            "disease_p_macro": sum(r["disease_p"] for r in recs) / len(recs),
            "disease_r_macro": sum(r["disease_r"] for r in recs) / len(recs),
            "coverage_macro": (sum(covs) / len(covs)) if covs else None,
            "n_cov_items": len(covs),
            "halluc_total": sum(r["halluc_count"] for r in recs),
        })
    return agg


def cmd_gold(set_names: list[str]) -> int:
    sysp = system_prompt()
    print(f"[gold] system prompt {len(sysp)} chars（SFT 首行现取）")
    all_pass = True
    for s in set_names:
        items = load_eval(s)
        recs = [score_item(it, gold_target(it)) for it in items]
        agg = aggregate(recs, s)
        fails = [r["id"] for r in recs if not r["fmt_ok"]]
        extra = []
        if s.startswith("neg"):
            neg_fails = [r["id"] for r in recs if not r["neg_hit"]]
            ok = agg["fmt_ok"] == agg["n"] and not neg_fails
            extra = [f"⑥无病命中 {agg['neg_hit']}/{agg['n']}"
                     f"（reason_ok 软读数 {agg['neg_reason_ok']}/{agg['n']}）",
                     f"⑥失败={neg_fails[:8]}"]
        elif s == "test-rob":
            rob_fails = [r["id"] for r in recs if not r["rob_hit"]]
            ok = agg["fmt_ok"] == agg["n"] and not rob_fails
            extra = [f"④ garbled {agg['rob_hit_garbled']}/{agg['n_garbled']}"
                     f" minimal {agg['rob_hit_minimal']}/{agg['n_minimal']}", f"④失败={rob_fails[:8]}"]
        else:
            cov_bad = [r["id"] for r in recs if r["coverage"] not in (None, 1.0)]
            hal_bad = [r["id"] for r in recs if r["halluc_count"]]
            ok = (agg["disease_f1_macro"] == 1.0 and agg["fmt_ok"] == agg["n"]
                  and agg["coverage_macro"] == 1.0 and agg["halluc_total"] == 0)
            extra = [f"①F1={agg['disease_f1_macro']:.4f} ③cov={agg['coverage_macro']:.4f}"
                     f"（{agg['n_cov_items']}/{agg['n']} 条有原子） ⑤幻觉={agg['halluc_total']}",
                     f"①≠1={fails[:0] or [r['id'] for r in recs if r['disease_f1'] != 1.0][:8]}"
                     f" ②失败={fails[:8]} ③<1={cov_bad[:8]} ⑤>0={hal_bad[:8]}"]
        all_pass &= ok
        print(f"[gold] {s}: ② {agg['fmt_ok']}/{agg['n']} · " + " · ".join(extra) + " → " + ("过" if ok else "败"))
    print("[gold] 五轴门：" + ("全满分 ✓ 允许跑模型" if all_pass else "未过 ✗ 禁止跑模型"))
    return 0 if all_pass else 1


# ---------------------------------------------------------------- prep / infer / score / table

def cmd_prep(set_names: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sysp = system_prompt()
    for s in set_names:
        items = load_eval(s)
        dst = OUT / f"infer-input-{s}.jsonl"
        assert not dst.exists(), f"证据只增不改，已存在：{dst}"
        dst.write_text(
            "".join(json.dumps({"id": it["id"], "messages": [
                {"role": "system", "content": sysp},
                {"role": "user", "content": it["user"]}]}, ensure_ascii=False) + "\n"
                for it in items), encoding="utf-8")
        print(f"[prep] {s}: {len(items)} 条 → {dst}")
    return 0


def cmd_infer(tag: str, model: Path | None, set_name: str, quant_bits: int | None, limit: int | None,
              batch: int, adapters: Path | None, out: Path, quant: str = "auto") -> int:
    out.mkdir(parents=True, exist_ok=True)
    src = out / f"infer-input-{set_name}.jsonl"
    if not src.exists():  # prep 输入模型无关：MED-B 复用 MED-A 既有渲染，不重复落证据
        src = REPO / "output" / "med-a" / f"infer-input-{set_name}.jsonl"
    assert src.exists(), f"缺推理输入，先跑 prep：{src}"
    result = out / f"{tag}-{set_name}.jsonl"
    assert not result.exists(), f"证据只增不改，已存在：{result}"
    if limit:  # 最小实验用：临时截短输入（不影响 infer-input 原件）
        tmp = out / f"infer-input-{set_name}.mini{limit}.jsonl"
        if not tmp.exists():
            tmp.write_text("".join(src.read_text(encoding="utf-8").splitlines(True)[:limit]), encoding="utf-8")
        src = tmp
        result = out / f"{tag}-{set_name}.mini{limit}.jsonl"
        assert not result.exists(), f"已存在：{result}"

    env = dict(os.environ)
    if adapters:  # ckpt 评测（b3 谱系）：模板/量化/思考开关从 args.json 恢复，不传 model/template
        args_json = (adapters / "args.json") if adapters.is_dir() else adapters.parent / "args.json"
        meta = json.loads(args_json.read_text(encoding="utf-8"))
        print(f"[infer] adapters={adapters}（args.json 恢复：template={meta.get('template')} "
              f"enable_thinking={meta.get('enable_thinking')} "
              f"quant={meta.get('quant_method')}-{meta.get('quant_bits')}）")
        cmd = [str(SWIFT), "infer", "--adapters", str(adapters),
               "--val_dataset", str(src), "--temperature", "0",
               "--max_new_tokens", "768", "--result_path", str(result),
               "--max_batch_size", str(batch)]
        if quant == "bf16":  # MED-E bf16 消口径列（D-08 增件·D1 开发）：显式覆盖 args.json 的量化恢复
            # swift 的 ckpt args 恢复只在"当前值为 None"时生效（base_args.load_args_from_ckpt），
            # quant_method 无 'none' 字面量 → 覆盖路径 = --load_args false 整体跳过恢复，
            # 基座/模板/思考开关改显式传（取自同一 args.json，保口径一致），不传 quant 即 bf16
            # device_map+max_memory（E2 首跑实测 2026-09-30）：单卡整装在常驻外 ~11GiB 下
            # caching_allocator_warmup 即 OOM；裸 auto 则 CPU 侧层留在 mmap 页（RSS 530MB），
            # 遇 swap 满 → 每前向一次缺页风暴（GPU 0%/CPU 闲/224s/条）。修复 = 显式预算
            # {"0": 9GiB, cpu: 40GiB} 把 CPU 侧层实体化进进程 RSS——加载机制修补，判分口径零改动
            cmd += ["--load_args", "false",
                    "--model", str(meta.get("model")),
                    "--template", str(meta.get("template")),
                    "--enable_thinking", "false" if not meta.get("enable_thinking") else "true",
                    "--device_map", "auto",
                    "--max_memory", BF16_MAX_MEMORY]  # int 键 0：JSON 串键会被 accelerate 拒（实测）
            print(f"[infer] quant 覆盖 bf16（load_args=false · device_map=auto · max_memory {BF16_MAX_MEMORY} · "
                  f"model={meta.get('model')} · template={meta.get('template')}）——14B bf16 > 16GB 显存，GPU+CPU offload")
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        assert model, "ckpt 评测用 --adapters，base 零样本需 <模型路径> 位置参数"
        # 模板按上游方法学仓的 b3 先例：4B-Instruct-2507（非思考档）qwen3_nothinking，其余 qwen3 关思考
        if "Instruct" in model.name:
            template_cmd = ["--template", "qwen3_nothinking"]
        else:
            template_cmd = ["--template", "qwen3", "--enable_thinking", "false"]
        cmd = [str(SWIFT), "infer", "--model", str(model), *template_cmd,
               "--val_dataset", str(src), "--temperature", "0",
               "--max_new_tokens", "768", "--result_path", str(result),
               "--max_batch_size", str(batch)]
        if quant_bits:
            cmd += ["--quant_method", "bnb", "--quant_bits", str(quant_bits)]
    print(f"[infer] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=REPO, env=env)
    print(f"[infer] {tag}/{set_name} 完成 {time.time() - t0:.0f}s → {result}", flush=True)
    return 0


def cmd_score(tag: str, set_name: str, out: Path) -> int:
    result = out / f"{tag}-{set_name}.jsonl"
    rows = [json.loads(ln) for ln in result.read_text(encoding="utf-8").splitlines() if ln.strip()]
    items = load_eval(set_name)
    assert len(rows) == len(items), f"行数不齐：infer={len(rows)} eval={len(items)}"
    recs = []
    for i, (row, it) in enumerate(zip(rows, items)):  # swift infer 保序（输出行无 id）
        resp = row.get("response") or ""
        if isinstance(resp, list):  # messages 形态兜底
            resp = resp[-1]["content"]
        think = re.search(r"<think>(.*?)</think>", resp, re.S)
        resp_clean = resp.replace(think.group(0), "", 1) if think else resp
        recs.append(score_item(it, resp_clean))
        recs[-1]["think_nonempty"] = bool(think and think.group(1).strip())
    agg = aggregate(recs, set_name)
    dst = out / f"score-{tag}-{set_name}.json"
    assert not dst.exists(), f"证据只增不改，已存在：{dst}"
    dst.write_text(json.dumps(
        {"tag": tag, "set": set_name, "aggregate": agg,
         "records": [{**r, "resp_head": (rows[i].get("response") or "")[:160]} for i, r in enumerate(recs)]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[score] {tag}/{set_name}: " + json.dumps(agg, ensure_ascii=False))
    print(f"[score] 证据 → {dst}")
    return 0


def cmd_table(out: Path) -> int:
    files = sorted(out.glob("score-*.json"))
    assert files, "无可聚合 score 文件"
    data = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    tags = sorted({d["tag"] for d in data})
    print("| 模型 | 集 | ①F1宏 | ①P | ①R | ②格式 | ③覆盖 | ⑤幻觉 | ④garbled | ④minimal | ⑥无病 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for tag in tags:
        for d in [x for x in data if x["tag"] == tag]:
            a = d["aggregate"]
            if d["set"].startswith("neg"):
                print(f"| {tag} | {d['set']} | — | — | — | {a['fmt_ok']}/{a['n']} | — | — | — |"
                      f" ⑥无病 {a['neg_hit']}/{a['n']} |")
            elif d["set"] == "test-rob":
                print(f"| {tag} | {d['set']} | — | — | — | {a['fmt_ok']}/{a['n']} | — | — |"
                      f" {a['rob_hit_garbled']}/{a['n_garbled']} | {a['rob_hit_minimal']}/{a['n_minimal']} |")
            else:
                cov = f"{a['coverage_macro']:.3f}" if a["coverage_macro"] is not None else "—"
                print(f"| {tag} | {d['set']} | {a['disease_f1_macro']:.3f} | {a['disease_p_macro']:.3f} |"
                      f" {a['disease_r_macro']:.3f} | {a['fmt_ok']}/{a['n']} | {cov} | {a['halluc_total']} | — | — |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gold").add_argument("sets", nargs="+", choices=SETS)
    sub.add_parser("prep").add_argument("sets", nargs="+", choices=SETS)
    p_inf = sub.add_parser("infer")
    p_inf.add_argument("tag"), p_inf.add_argument("model", type=Path, nargs="?", default=None)
    p_inf.add_argument("set", choices=SETS)
    p_inf.add_argument("--adapters", type=Path, default=None,
                       help="ckpt 评测（b3 谱系）：模板/量化从 args.json 恢复，与 --model 二选一")
    p_inf.add_argument("--quant_bits", type=int, default=None)
    p_inf.add_argument("--quant", choices=["auto", "bf16"], default="auto",
                       help="auto=args.json 自动恢复；bf16=显式覆盖量化（load_args false+显式基座，MED-E 消口径列）")
    p_inf.add_argument("--limit", type=int, default=None)
    p_inf.add_argument("--out", type=Path, default=OUT)
    p_inf.add_argument("--batch", type=int, default=None,
                       help="缺省：adapters 模式 1（贴边），base 模式 16（MED-A 行为）")
    p_sco = sub.add_parser("score")
    p_sco.add_argument("tag"), p_sco.add_argument("set", choices=SETS)
    p_sco.add_argument("--out", type=Path, default=OUT)
    p_tab = sub.add_parser("table")
    p_tab.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.cmd == "gold":
        return cmd_gold(args.sets)
    if args.cmd == "prep":
        return cmd_prep(args.sets)
    if args.cmd == "infer":
        if args.quant == "bf16":
            assert args.adapters, "--quant bf16 只在 adapters 模式有意义（base 模式不传 --quant_bits 即 bf16）"
        batch = args.batch if args.batch is not None else (1 if args.adapters else 16)
        return cmd_infer(args.tag, args.model.resolve() if args.model else None, args.set,
                         args.quant_bits, args.limit, batch, args.adapters, args.out, args.quant)
    if args.cmd == "score":
        return cmd_score(args.tag, args.set, args.out)
    return cmd_table(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
