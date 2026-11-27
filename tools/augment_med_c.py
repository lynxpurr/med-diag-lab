#!/usr/bin/env python3
"""MED-C 合成增强器（augment.py 谱系拷贝改造，LAB-E R6 泄题防线的本仓版）.

输入只读：data/med0/swift-train-f1.jsonl（1154，冠军臂训练集——变体全部源自其内部，
评测集 val-reg/test-ood/test-rob/neg/negn 一字不入；无病臂为新生成，另有与 30 条
评测素材的文本相似度上限断言）。输出（data/medc/）：
  - aug-garbled.jsonl / aug-minimal.jsonl / aug-neg.jsonl   三臂增强变体（36/臂）
  - swift-train-c{g,m,n}.jsonl   臂合并集 = f1 1154 原序 + aug（LAB-E 臂合并约定）
  - aug-manifest.json            选样规则/配额/断言汇总（fidelity 断言全绿才落盘）

三臂（PLAN §4 MED-C 校准块：保持+防回退波，目标行为 = MED-B 冠军已存活的形态）：
  garbled 36  乱码注入（build_dataset.garble 同函数同字母表，aug id 独立偏移）
              → 教规则 5：数据异常块 + 分条引用具体乱码字符（冠军 14B 实测形态）。
  minimal 36  = M1 仅性别/年龄/主诉 18（minimize 同函数）→ 教规则 6 推测：源疾病列表 +
              主诉引用 + 补充信息建议（冠军 minimal 全走推测出口，不走缺少信息出口）
            + M2 字段删除保格式 18（删 辅助检查/体格检查，护栏 = 删后长度 ≥60% 源）
              → 教"证据减半仍诊断，但依据只引在场证据"（原子级过滤源依据）。
  neg     36  无病/阴性化（形态 A14 + B16 + C6，人口学锚定自 train + 健康模板体查，
              数值锚 WS/T 404/405 + TSH 0.27-4.2）→ 教规则 4 + 防形态 B 边界漂移
              （neg-013 冠军唯一 miss；B 配额最高）。与 30 条评测素材文本相似度
              quick_ratio < 0.90（防近重复泄题）。

fidelity 断言（任一不过整体退出非零，augment.py 对账谱系）：
  通用：目标经 run_eval_med.score_item 复判（评测器是判分单一事实源）——fmt_ok 必真；
        依据数值原子全部在场（幻觉轴教学侧零编造：halluc==0 / 原子子集断言）。
  garbled：非换行字符 100% ∈ 乱码字母表 ∧ 换行结构保持 ∧ rob_hit。
  minimal：M1 仅三行 ∧ rob_hit（推测出口）；M2 长度 ≥60% ∧ F1==1.0 ∧ 源依据原子级过滤。
  neg：检验值全部落锚定区间 ∧ 现症阳性扫描零命中（否定/既往/已愈护栏）∧ neg_hit
       ∧ 与 30 条评测素材相似度 <0.90 ∧ id 前缀 negt- 与评测 neg-/negn- 不相交。

确定性：选样全部稳定排序（id 序）；数值抽样 random.Random(20260925) 单实例顺序消费。
用法：python3 tools/augment_med_c.py   （在仓根跑；重复执行会因证据只增不改而拒绝）
"""
from __future__ import annotations

import difflib
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from build_dataset import GARBLE_ALPHABET, build_target, garble, minimize  # noqa: E402
from normalize import parse_diseases_tag  # noqa: E402
from run_eval_med import (  # noqa: E402
    atom_present, gold_target, keyword_atoms, score_item, system_prompt, value_atoms, _norm,
)

F1 = REPO / "data" / "med0" / "swift-train-f1.jsonl"
MEDC = REPO / "data" / "medc"
SEED = 20260925
N_GARBLED = N_MINIMAL = 36  # 各臂 36（30-40 带内取中）
M1_QUOTA, M2_QUOTA = 18, 18
NEG_QUOTA = {"A": 14, "B": 16, "C": 6}  # B（自限已缓解）最高——neg-013 边界对症

# low 变体（R5' 回退分支的预注册响应：合成配比下调重跑一次，只重跑 cn 臂——cg/cm 无独占
# 教学载荷且其失败轴=无病回退，低配比无修复路径；场景库前 N 配额，值抽样流独立非子集）
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "full"
if VARIANT == "low":
    N_NEG = 12
    NEG_QUOTA = {"A": 4, "B": 6, "C": 2}
elif VARIANT == "full":
    N_NEG = 36
else:
    raise SystemExit(f"未知变体 {VARIANT}（full | low）")


def load_f1() -> list[dict]:
    rows = [json.loads(ln) for ln in F1.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for r in rows:
        user, assistant = r["messages"][1]["content"], r["messages"][2]["content"]
        m = re.search(r"<diseases>(.*?)</diseases>", assistant, re.S)
        reasons = re.findall(r"(?m)^\s*\d+[\.、]\s*(.+)$", assistant)
        # parse_diseases_tag 不剥空白（评测侧 prf 会归一化），训练目标须干净名——此处剥净
        diseases = [d.strip() for d in (parse_diseases_tag(assistant) or []) if d.strip()]
        out.append({"id": r["id"], "user": user, "diseases": diseases,
                    "reasons": reasons, "n_reasons_raw": len(reasons),
                    "tail": assistant[m.end():] if m else ""})
    return out


def gate(rec: dict, want: str) -> None:
    """评测器复判门：fmt_ok 恒真 + 按 want 断言语义轴。gold 渲染走 gold_target（单一事实源，
    garbled 特例块=数据异常）。注意训练侧 target 用各臂自己的构造（如 garbled 用 数据异常 块），
    此门断言的正是那个 target——调用方 rec.gold 按臂就位."""
    scored = score_item(rec, gold_target(rec))
    assert scored["fmt_ok"], f"fmt 门挂: {scored}"
    if want == "garbled":
        assert scored["rob_hit"], f"garbled rob_hit 挂: {scored}"
    elif want == "minimal":
        assert scored["rob_hit"], f"minimal rob_hit 挂: {scored}"
    elif want == "f1":
        assert scored["disease_f1"] == 1.0, f"F1≠1: {scored}"
        assert scored["halluc_count"] == 0, f"幻觉原子: {scored['halluc_atoms']}"
    elif want == "neg":
        assert scored["neg_hit"], f"neg_hit 挂: {scored}"


def reasons_atoms_in(user: str, reasons: list[str]) -> bool:
    """无病臂原子在场断言（轴⑥分支不算幻觉，教学侧自证：依据引用值全在输入）."""
    u_raw, u_norm = user, _norm(user)
    for r in reasons:
        for tok in value_atoms(r):
            if not atom_present(tok, u_raw, u_norm):
                return False
    return True


# ---------------------------------------------------------------- 臂 1：garbled

def build_garbled(src: list[dict], sysp: str) -> list[dict]:
    out, seen = [], set()
    for s in sorted(src, key=lambda x: x["id"]):  # 稳定排序选样，体量 ≥300 字符保证乱码信息量
        if len(out) == N_GARBLED:
            break
        if len(s["user"]) < 300 or s["id"] in seen:
            continue
        aug_id = f"augg-{len(out)+1:03d}"
        garbled = garble(s["user"], aug_id)
        # 断言：换行结构保持 ∧ 非换行字符全落字母表（无残留可读信息）
        assert garbled.count("\n") == s["user"].count("\n")
        assert all(ch == "\n" or ch in GARBLE_ALPHABET for ch in garbled)
        lines = [ln for ln in garbled.splitlines() if len(ln) >= 6]
        cands = [ln[i:i + 4] for ln in lines[:6] for i in range(0, min(4, len(ln) - 3))]
        assert cands, "乱码行不足（≥6 字符行 ≥1）"
        q1 = cands[0]
        q2 = next((q for q in cands[1:] if q != q1), None)
        assert q2, "乱码子串全同（信息量不足，拒）"
        reasons = [
            f"检查到输入的病历条目包含大量乱码和无意义字符，如‘{q1}’、‘{q2}’等，这些字符无法解析为有效的医疗信息。",
            "由于乱码的存在，无法提取出患者的性别、年龄、主诉、现病史、既往史、体格检查、辅助检查等关键信息，无法进行疾病诊断。",
            "病历内容不符合正常病历文本的规范，属于与病情无关的编造数据，判定为数据异常。",
        ]
        rec = {"id": aug_id, "user": garbled, "split": "test_rob",
               "condition_kind": "garbled", "expected_category": "数据异常",
               "gold": {"diseases": s["diseases"], "reasons": reasons}}  # diseases 记源（rob 不判 ①）
        gate(rec, "garbled")
        assert not value_atoms(" ".join(reasons)), "garbled 依据不应含数值原子"
        out.append({"messages": [{"role": "system", "content": sysp},
                                 {"role": "user", "content": garbled},
                                 {"role": "assistant", "content": build_target(["数据异常"], reasons)}],
                    "id": aug_id, "form": "aug-garbled",
                    "aug_meta": {"mode": "garbled", "source_id": s["id"],
                                 "source_len": len(s["user"]), "quotes": [q1, q2]}})
        seen.add(s["id"])
    assert len(out) == N_GARBLED, f"garbled 候选不足: {len(out)}"
    return out


# ---------------------------------------------------------------- 臂 2：minimal

def build_minimal(src: list[dict], sysp: str) -> tuple[list[dict], dict]:
    out = []
    # M1 仅性别/年龄/主诉 → 规则 6 推测（冠军实测出口：给疾病列表 + 主诉引用 + 建议补充）
    for s in sorted(src, key=lambda x: x["id"]):
        if len([r for r in out if r["aug_meta"]["mode"] == "minimal-m1"]) == M1_QUOTA:
            break
        if not (1 <= len(s["diseases"]) <= 2) or len(s["reasons"]) < 2:
            continue
        mini = minimize(s["user"])
        lines = mini.splitlines()
        if len(lines) != 3 or not all(re.match(r"^(性别|年龄|主诉)：.+$", ln) for ln in lines):
            continue
        chief = re.search(r"^主诉：(.+)$", mini, re.M).group(1)
        if len(chief) < 6:
            continue
        sex = re.search(r"^性别：(.+)$", mini, re.M).group(1)
        age = re.search(r"^年龄：(\d+)", mini, re.M).group(1)
        d_join = "与".join(s["diseases"])
        reasons = [
            f"病历仅提供性别、年龄与主诉“{chief}”，信息有限，以下为根据有限信息作出的推测。",
            f"根据主诉“{chief}”，首先考虑{d_join}的可能。",
            f"现有信息不足以明确诊断，建议补充现病史、体格检查及辅助检查资料（患者为{age}岁{sex}）。",
        ]
        rec = {"id": f"augm1-{len(out)+1:03d}", "user": mini, "split": "test_rob",
               "condition_kind": "minimal", "expected_category": "推测或缺少关键诊断信息",
               "gold": {"diseases": s["diseases"], "reasons": reasons}}
        gate(rec, "minimal")
        assert reasons_atoms_in(mini, reasons), "M1 依据原子不在场"
        out.append({"messages": [{"role": "system", "content": sysp},
                                 {"role": "user", "content": mini},
                                 {"role": "assistant", "content": build_target(s["diseases"], reasons)}],
                    "id": rec["id"], "form": "aug-minimal",
                    "aug_meta": {"mode": "minimal-m1", "source_id": s["id"], "chief": chief}})
    assert sum(1 for r in out if r["aug_meta"]["mode"] == "minimal-m1") == M1_QUOTA

    # M2 字段删除保格式 → 源疾病 + 原子级过滤后的源依据（只引在场证据）
    DELETE_ORDER = [("辅助检查",), ("体格检查",), ("辅助检查", "体格检查")]
    MARKER = {"辅助检查": "辅助检查|辅查|检查示|检验|化验|实验室|血常规|尿常规|便常规|心电图|超声|X线|胸片|CT|MRI|造影|血沉|生化",
              "体格检查": "体格检查|查体|触诊|视诊|叩诊|听诊|专科查体|体格"}
    m2_stat = {"tried": 0}
    for s in sorted(src, key=lambda x: x["id"]):
        if sum(1 for r in out if r["aug_meta"]["mode"] == "minimal-m2") == M2_QUOTA:
            break
        if len(s["diseases"]) < 1 or len(s["reasons"]) < 2:
            continue
        for fields in DELETE_ORDER:
            if not all(re.search(rf"^{f}：..+$", s["user"], re.M) for f in fields):
                continue  # 待删字段须非空在场
            reduced = "\n".join(ln for ln in s["user"].splitlines()
                                if not any(ln.startswith(f + "：") for f in fields))
            if len(reduced) < 0.60 * len(s["user"]):
                continue  # 护栏：删后长度 ≥60% 源（中文文本工程护栏，playbook §2）
            r_norm = _norm(reduced)
            kept = []
            for r in s["reasons"]:
                if any(re.search(MARKER[f], r) for f in fields):
                    continue  # 引用已删字段的定性句剔除
                if all(atom_present(t, reduced, r_norm) for t in value_atoms(r)) and \
                   all(k in reduced for k in keyword_atoms(r)):
                    kept.append(r)
            if not kept:
                continue
            m2_stat["tried"] += 1
            rec = {"id": f"augm2-{m2_stat['tried']:03d}", "user": reduced, "split": "train",
                   "gold": {"diseases": s["diseases"], "reasons": kept}}
            gate(rec, "f1")
            out.append({"messages": [{"role": "system", "content": sysp},
                                     {"role": "user", "content": reduced},
                                     {"role": "assistant", "content": build_target(s["diseases"], kept)}],
                        "id": rec["id"], "form": "aug-minimal",
                        "aug_meta": {"mode": "minimal-m2", "source_id": s["id"],
                                     "deleted": list(fields), "kept_reasons": len(kept),
                                     "src_reasons": len(s["reasons"])}})
            break
    assert sum(1 for r in out if r["aug_meta"]["mode"] == "minimal-m2") == M2_QUOTA, "M2 候选不足"
    return out, m2_stat


# ---------------------------------------------------------------- 臂 3：neg（无病）

# 数值锚（WS/T 404/405-2012 成人区间 + TSH 0.27-4.2 取窄带留安全边际；与 neg 评测素材同锚）
RANGES = {
    "wbc": (4.0, 9.5), "neut": (45, 70), "plt": (150, 340),
    "fpg": (4.2, 5.8), "tc": (3.5, 5.1), "tg": (0.6, 1.6),
    "alt": (12, 45), "ast": (15, 38), "tsh": (0.5, 3.8),
    "rbc_m": (4.3, 5.7), "rbc_f": (3.8, 5.0), "hb_m": (130, 170), "hb_f": (115, 145),
    "cr_m": (60, 95), "cr_f": (45, 80),
    "t": (36.1, 36.9), "p": (62, 88), "r": (16, 19), "bp_sys": (105, 128), "bp_dia": (68, 82),
}
# 现症阳性扫描表（小句否定辖域/同句已愈/既往标记 三护栏；unnegated 命中即挂）
POSITIVE_TOKENS = ("压痛", "反跳痛", "肌紧张", "包块", "肿大", "水肿", "浮肿", "黄疸", "发绀",
                   "苍白", "杂音", "啰音", "皮疹", "出血点", "静脉曲张", "溃疡", "糜烂",
                   "结节", "充血", "渗出", "压痛阳性", "肌阵挛", "贫血貌")
RESOLVED_MARK = re.compile(r"已愈|痊愈|治愈|已缓解|缓解|消失|消退|恢复|已吸收|已愈合|复查正常")
PAST_MARK = re.compile(r"(?:\d+(?:\.\d+)?(?:天|日|周|月|年)|数天|数日|半月|一月|前一?段时间)前|既往|曾有|病史")
CLAUSE_BREAK = re.compile(r"[。；;，、：\n（]")


def scan_positives(text: str) -> list[str]:
    """unnegated 现症阳性命中。中文并列否定辖域跨顿号/及（无压痛及反跳痛）——
    判据 = 当前小句（token 前）含否定词，或同句带已愈标记，或紧邻前文带既往时间标记."""
    hits = []
    for tok in POSITIVE_TOKENS:
        for m in re.finditer(re.escape(tok), text):
            seg = m.start()
            clause = CLAUSE_BREAK.split(text[:seg])[-1]
            if re.search(r"[无未非]", clause) or clause.endswith(("原", "曾", "既往")):
                continue  # 小句否定辖域 / 指涉性既往（"原皮疹处无痕迹"）
            sent = re.split(r"[。；;]", text[seg:])[0]  # 同句（向后取到句末）
            if RESOLVED_MARK.search(sent) or RESOLVED_MARK.search(clause):
                continue  # 同句已愈（形态 B 既往症叙事，标记可在 token 前或后）
            if PAST_MARK.search(text[max(0, seg - 14):seg]):
                continue  # 既往时间标记
            hits.append(tok)
    return hits


def vitals(rng: random.Random, idx: int) -> dict:
    return {"t": f"{rng.uniform(*RANGES['t']):.1f}", "p": str(rng.randint(*RANGES['p'])),
            "r": str(rng.randint(*RANGES['r'])),
            "bp": f"{rng.randint(*RANGES['bp_sys'])}/{rng.randint(*RANGES['bp_dia'])}"}


def labs(rng: random.Random, sex: str, panel: int) -> str:
    """检验面板渲染。句式与评测素材（neg/negn 草稿）刻意不同（泄题防线：叙事块 ratio<0.80），
    数值全部落 WS/T 锚定区间（RANGES）。"""
    wbc = f"{rng.uniform(*RANGES['wbc']):.1f}"; neut = str(rng.randint(*RANGES["neut"]))
    rbc_key, hb_key, cr_key = ("rbc_m", "hb_m", "cr_m") if sex == "男" else ("rbc_f", "hb_f", "cr_f")
    rbc = f"{rng.uniform(*RANGES[rbc_key]):.2f}"; hb = str(rng.randint(*RANGES[hb_key]))
    plt = str(rng.randint(*RANGES["plt"])); fpg = f"{rng.uniform(*RANGES['fpg']):.1f}"
    cr = str(rng.randint(*RANGES[cr_key])); tc = f"{rng.uniform(*RANGES['tc']):.1f}"
    tg = f"{rng.uniform(*RANGES['tg']):.1f}"; alt = str(rng.randint(*RANGES["alt"]))
    ast = str(rng.randint(*RANGES["ast"])); tsh = f"{rng.uniform(*RANGES['tsh']):.2f}"
    base = (f"WBC {wbc}×10⁹/L、中性粒细胞比例 {neut}%、RBC {rbc}×10¹²/L、"
            f"Hb {hb}g/L、PLT {plt}×10⁹/L")
    urine = "外观清亮，蛋白、葡萄糖、潜血、酮体及镜检白细胞均阴性"
    chem = (f"空腹血糖 {fpg}mmol/L、总胆固醇 {tc}mmol/L、甘油三酯 {tg}mmol/L、"
            f"血肌酐 {cr}μmol/L、ALT {alt}U/L、AST {ast}U/L")
    thyroid = f"TSH {tsh}mIU/L"
    panels = {0: f"血常规：{base}。尿分析：{urine}。血生化：{chem}。",
              1: f"血常规：{base}。尿分析：{urine}。甲状腺功能：{thyroid}。心电图：窦性心律，大致正常。",
              2: f"血常规：{base}。尿分析：{urine}。血生化：{chem}。心电图：窦性心律，大致正常。"
                 "腹部超声：肝、胆、胰、脾、双肾声像图未见异常。",
              3: f"血常规：{base}。尿分析：{urine}。甲状腺功能：{thyroid}。胸部X线：双肺及心影未见异常。"}
    return panels[panel % 4]


def pe_normal(v: dict, variant: int) -> str:
    """正常体查渲染，三变体轮换；句式与评测素材草稿刻意不同（如 体温/脉搏 全称、
    腹柔软、浮肿 等措辞与语序重组——语义必须同（正常体查），句法刻意偏离）."""
    head = (f"体温{v['t']}℃，脉搏{v['p']}次/分，呼吸{v['r']}次/分，血压{v['bp']}mmHg。"
            "神志清晰，发育营养正常，自动体位，查体配合。")
    bodies = [
        "皮肤巩膜未见黄染，周身浅表淋巴结无肿大。颈软，气管居中，甲状腺不大。双肺呼吸音清，"
        "未闻啰音。心律齐，心前区未闻及病理性杂音。腹柔软，肝脾未触及，无压痛反跳痛。"
        "双下肢无浮肿。生理反射正常引出，病理反射未引出。",
        "全身皮肤黏膜无黄染，浅表淋巴结不大，头颈部检查无异常发现。两肺叩诊清音，呼吸音清"
        "晰，未闻及干湿啰音。心界不大，节律齐，各瓣膜区无病理性杂音。腹软无压痛，肝脾肋下"
        "未及。脊柱四肢活动自如，双下肢无肿。膝腱反射存在，锥体束征阴性。",
        "一般状况好，皮肤无黄染及出血点，表浅淋巴结未扪及肿大。颈项软，甲状腺无结节。肺部"
        "呼吸音正常。心率齐，未闻及杂音。腹部平坦柔软，无压痛，未触及包块。四肢关节无红肿"
        "及活动受限，双无水肿。神经系统检查未见异常体征。",
    ]
    return head + bodies[variant % 3]


# 形态 A：全正常体检 14 场景——场景类型与评测素材不相交（评测用 年度/单位/入职/绝经后/
# 在校/退休后/婚前 体检；本表全部另选），教"各色体检场景 → 无"的泛化而非场景记忆。
A_SCENARIOS = [
    "机动车驾驶证换证体检", "健康保险投保前体检", "教师资格认定体检", "公务员录用体检",
    "研究生入学复查体检", "餐饮行业健康证明办理体检", "无偿献血前健康筛查", "运动员年度注册体检",
    "普通高校招生体检", "出国留学前体检", "职业健康监护例行体检", "定期健康随访体检",
    "家族疾病史咨询伴健康体检", "康养机构入住前健康评估",
]
A_AGE = [(22, 60), (22, 60), (22, 50), (22, 40), (20, 35), (20, 50), (18, 55), (16, 40),
         (15, 22), (16, 40), (20, 60), (30, 70), (25, 65), (60, 85)]
# 形态 B：自限/已愈复查 16 场景（neg-013 边界对症族：踝扭伤/上感/胃肠炎 三族保留——
# 边界类迁移正是教学点；侧别/时程/机制/措辞与评测素材逐条偏离，主诉句式不同构）
B_SCENARIOS = [
    # (主诉, 病名, 既往经过, 现况, 复查, 年龄段)
    ("咽痛消退后复查", "上呼吸道感染",
     "一周前着凉后咽干、咽痛并伴低热，居家休息数日后体温恢复正常，咽痛随之缓解",
     "目前无发热，无咳嗽及咽痛，咽部黏膜色泽正常", "复查血常规各值均在参考区间", (8, 70)),
    ("腹泻停止后复查", "急性胃肠炎",
     "四天前进食剩菜后阵发腹痛、排水样便数次，门诊予口服补液，当日腹泻即止",
     "现排便成形，腹部触诊柔软无压痛", "粪便常规未见红白细胞", (15, 65)),
    ("左踝扭伤后复诊", "踝部软组织扭伤",
     "两周前打球时左踝内翻扭伤，当晚踝部肿痛，冰敷并制动一周后肿痛消退",
     "现左踝外形无肿胀，行走及上下楼梯无疼痛，关节各方向活动不受限",
     "左踝X线片示骨质与关节间隙无异常", (12, 55)),
    ("腰痛缓解后复查", "急性腰肌扭伤",
     "六天前提重物后腰部酸痛、弯腰受限，卧床休息两天后酸痛渐轻",
     "现腰部肌肉无压痛，弯腰及转身活动自如", "腰椎正侧位片未见骨质病变", (18, 65)),
    ("前臂淤青吸收后复查", "皮肤软组织挫伤",
     "两周前手臂磕碰桌角后局部青紫一片，未做特殊处理",
     "现局部青紫已吸收褪净，触诊无压痛", "软组织无肿胀，肢体活动正常", (10, 70)),
    ("咽部不适缓解后复查", "急性咽炎",
     "数日前咽部干痒刺痛，多饮温水并注意休息后症状消失",
     "现咽后壁黏膜无充血，咽部无不适", "口咽部检查未见异常", (12, 65)),
    ("眼睑小肿物消退后复查", "睑腺炎（麦粒肿）",
     "十日前左眼睑缘出现小肿物伴微痛，自行热敷数日后肿物缩小至消失",
     "现眼睑无红肿、未触及结节，睑缘无分泌物", "眼科裂隙灯检查未见异常", (10, 60)),
    ("烫伤愈合后复查", "前臂浅表烫伤",
     "两周前做饭时前臂内侧被热油溅伤，创面起小水疱，门诊换药两次",
     "现创面已上皮化愈合，痂皮脱落，无渗液", "创面愈合良好，无感染征象", (15, 65)),
    ("鼻塞消失后复查", "急性鼻炎",
     "一周前气温骤降后鼻塞、流清涕，未服药，三日后自行缓解",
     "现双侧鼻腔通气好，黏膜无充血，鼻甲无肿大", "前鼻镜检查未见异常", (10, 65)),
    ("咳嗽消失后复查", "急性支气管炎",
     "两周余前咳嗽、咳少量白痰，服止咳化痰药物后咳嗽渐止",
     "现无咳嗽咳痰，双肺听诊呼吸音清且对称", "复查胸片两肺纹理在正常范围", (15, 70)),
    ("拔牙后创口复查", "拔牙创愈合期",
     "一周前于口腔科拔除阻生齿，术后按医嘱护理",
     "现创口无出血、无疼痛，张口度正常", "口腔检查创口愈合良好，无红肿渗出", (18, 60)),
    ("皮肤擦伤愈合后复查", "膝部皮肤擦伤",
     "十日前骑车摔倒擦伤右膝，自行清洁消毒处理",
     "现擦伤处结痂已脱落，皮肤完整", "局部无红肿、无渗出", (10, 60)),
    ("甲沟红肿消退后复查", "甲沟炎",
     "两周前右足第一趾甲缘红肿疼痛，局部消毒处理并正确修剪趾甲后消退",
     "现甲缘无红肿压痛，趾甲生长顺", "足部检查未见异常", (15, 65)),
    ("耳痛痊愈后复查", "急性中耳炎",
     "两周前感冒后左耳疼痛，门诊予滴耳液，三日后耳痛消失",
     "现无耳痛耳鸣，鼓膜完整、光泽正常", "耳镜复查未见异常", (8, 60)),
    ("皮疹消退后复查", "虫咬性皮炎",
     "一周前户外活动后小腿散在红色小丘疹伴痒，外用洗剂后疹退",
     "现皮肤无新发皮疹，原皮疹处无痕迹", "皮肤检查未见活动性皮损", (10, 60)),
    ("接种后反应消退随访", "疫苗接种后一过性局部反应",
     "三天前接种疫苗后注射局部轻微红肿伴一过性低热，次日热退、红肿渐消",
     "现注射处无红肿压痛，体温正常", "局部检查未见异常", (18, 70)),
]
# 形态 C：健康咨询无异常主诉 6 场景（与评测 C 族：疫苗前/备孕/参军/马拉松 不相交）
C_SCENARIOS = [
    ("体重管理咨询", "近半年体重平稳，希望评估代谢状况，无多饮多食多尿，无乏力", "体脂与腰围在正常范围", (20, 60)),
    ("睡眠情况咨询", "入睡偏慢月余，白天精神尚可，无夜间憋醒、打鼾及肢体抽动", "查体未见阳性体征", (22, 65)),
    ("开始规律运动前评估", "既往少动，拟开始慢跑锻炼，无胸痛胸闷、心悸及头晕史", "心肺体检未见异常", (18, 65)),
    ("营养与膳食咨询", "素食两年，希望评估营养状况，无消瘦、无脱发、无乏力", "血红蛋白等营养指标均在参考区间", (20, 60)),
    ("颈肩不适咨询", "既往伏案工作后颈肩发僵，调整工位并规律锻炼后已消失，现无任何不适，来咨询预防保健建议", "颈椎活动度正常，压颈试验阴性", (22, 60)),
    ("体检结果咨询", "上月体检报告显示数项临界值，本次来院复查", "复查各项指标均在参考区间", (30, 70)),
]


def demo_anchors(src: list[dict]) -> list[dict]:
    """人口学锚定（D 形态谱系）：train 病历的 (性别,年龄) 去重顺序取，附源 id/疾病."""
    seen, out = set(), []
    for s in sorted(src, key=lambda x: x["id"]):
        m1 = re.search(r"^性别：(男|女)$", s["user"], re.M)
        m2 = re.search(r"^年龄：(\d{1,3})$", s["user"], re.M)
        if not (m1 and m2):
            continue
        key = (m1.group(1), int(m2.group(1)))
        if key in seen:
            continue
        seen.add(key)
        out.append({"sex": key[0], "age": key[1], "src": s["id"], "src_d": s["diseases"][0] if s["diseases"] else "?"})
        if len(out) >= 120:
            break
    return out


def build_neg(src: list[dict], sysp: str, eval_users: list[str]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    anchors = demo_anchors(src)
    used: set[tuple] = set()

    def take_anchor(lo: int, hi: int) -> dict:
        for a in anchors:  # 稳定顺序首个落段未用锚
            if (a["sex"], a["age"]) not in used and lo <= a["age"] <= hi:
                used.add((a["sex"], a["age"]))
                return a
        raise AssertionError(f"锚不足 [{lo},{hi}]")

    def assert_values_in_range(text: str) -> None:
        for m in re.finditer(r"(WBC|PLT|TSH) ([\d.]+)", text):
            key = {"WBC": "wbc", "PLT": "plt", "TSH": "tsh"}[m.group(1)]
            v = float(m.group(2))
            lo, hi = RANGES[key]
            assert lo <= v <= hi, f"{m.group(1)} {v} 出锚 {lo}-{hi}"

    def narrative(user: str) -> str:
        chief = re.search(r"^主诉：(.+)$", user, re.M).group(1)
        hpi = re.search(r"^现病史：(.+)$", user, re.M).group(1)
        return chief + "\n" + hpi

    def no_eval_leak(user: str) -> None:
        """三层防线：主诉非逐字同（评测主诉全文不含我主诉）∧ 全文 ratio<0.85
        ∧ 叙事块（主诉+现病史——条目身份所在，体查/检验是体裁样板）ratio<0.80.
        quick_ratio 上界先筛提速，ratio() 实测判据（体裁共享致底线 ~0.76-0.81，
        近重复判据只对叙事块收紧）."""
        narr = narrative(user)
        for e in eval_users:
            e_narr = narrative(e)
            assert narr not in e, f"叙事与评测素材重合：{narr[:30]}"
            sm_all = difflib.SequenceMatcher(None, user, e)
            if sm_all.quick_ratio() < 0.85:
                continue
            assert sm_all.ratio() < 0.85, f"全文相似度 {sm_all.ratio():.2f} ≥0.85"
            assert difflib.SequenceMatcher(None, narr, e_narr).ratio() < 0.80, \
                f"叙事块相似度 ≥0.80：{narr[:24]}…"

    out, rejects = [], []
    plans = ([("A", s) for s in zip(A_SCENARIOS, A_AGE)]
             + [("B", s) for s in B_SCENARIOS]
             + [("C", s) for s in C_SCENARIOS])
    if VARIANT == "low":  # 场景库前 N 配额（B 优先保边界载荷：B 前 6 全保留）
        by_form = {f: [p for p in plans if p[0] == f] for f in "ABC"}
        plans = sum(([p for p in by_form[f] if p[0] == f][:NEG_QUOTA[f]] for f in "ABC"), [])
    counters = {"A": 0, "B": 0, "C": 0}
    idx = 0
    for form, scen in plans:
        counters[form] += 1
        a = take_anchor(scen[-1][0], scen[-1][1])
        v = vitals(rng, idx)
        panel = labs(rng, a["sex"], idx)
        pe = pe_normal(v, idx)
        if form == "A":
            purpose = scen[0]
            chief = purpose
            history = (f"因{purpose}就诊。既往体健，近期无发热、咽痛、咳嗽、腹泻等症状，"
                       "进食与睡眠尚可，大小便如常，体重无明显变化。")
            reasons = [
                f"就诊目的为{purpose}，无任何症状主诉，生命体征正常（体温{v['t']}℃，脉搏{v['p']}次/分，血压{v['bp']}mmHg）。",
                "全身体格检查各系统未见阳性体征。",
                "辅助检查各项数值均在参考区间，心电图等检查无异常发现。",
                "综合以上判断为健康状态，无现患疾病。",
            ]
        elif form == "B":
            chief, cond, past, now, recheck, _ = scen
            history = f"{past}，当时诊断考虑{cond}。{now}，{recheck}。"
            reasons = [
                f"{cond}病史明确（{past.split('，')[0]}），经休息或简单处理后症状已完全消退。",
                f"目前{('，'.join(now.split('，')[1:]) if '，' in now else now)}；本次复查体温{v['t']}℃、脉搏{v['p']}次/分、血压{v['bp']}mmHg，均在正常范围。",
                f"{recheck}，无活动性病变证据。",
                f"{cond}属自限性或已痊愈疾病，当前不构成现患疾病，判断为身体健康。",
            ]
        else:
            topic, note, finding, _ = scen
            chief = topic
            history = f"因{topic}来诊。{note}，无其他不适主诉。"
            reasons = [
                f"来诊目的为{topic}，属健康咨询，无发热、疼痛等疾病症状。",
                f"体格检查与辅助检查无阳性发现：{finding}。",
                "无器质性疾病证据，判断为健康状态。",
            ]
        user = (f"性别：{a['sex']}\n年龄：{a['age']}\n主诉：{chief}\n现病史：{history}\n"
                f"既往史：平素健康，无肝炎、结核等传染病史，无手术外伤史，无输血史。\n"
                f"个人史：生长于原籍，无疫区旅居史，不吸烟不饮酒，无有毒有害物质接触史。\n"
                f"过敏史：否认药物及食物过敏史。\n"
                f"生育史：{'已育' if a['age'] >= 26 else '未生育'}\n"
                f"婚育史：{'已婚已育' if a['age'] >= 28 else '未婚'}\n"
                f"流行病史：无传染病患者接触史，无疫水接触史。\n"
                f"体格检查：{pe}\n辅助检查：{panel}")
        hits = scan_positives(user)
        assert not hits, f"现症阳性扫描命中 {hits}（id 拟 negt-{form}{counters[form]:02d}）"
        assert_values_in_range(user)
        assert reasons_atoms_in(user, reasons), "依据数值原子不在场"
        no_eval_leak(user)
        rec = {"id": f"negt-{form}{counters[form]:02d}", "user": user, "split": "test_rob",
               "condition_kind": "negative", "expected_category": "无",
               "gold": {"diseases": ["无"], "reasons": reasons}}
        gate(rec, "neg")
        assert 400 <= len(user) <= 2200, f"长度出带 {len(user)}"
        out.append({"messages": [{"role": "system", "content": sysp},
                                 {"role": "user", "content": user},
                                 {"role": "assistant", "content": build_target(["无"], reasons)}],
                    "id": rec["id"], "form": f"negt-{form}",
                    "aug_meta": {"mode": f"neg-{form}", "anchor_src": a["src"],
                                 "anchor_src_disease": a["src_d"], "sex": a["sex"], "age": a["age"]}})
        idx += 1
    assert len(out) == N_NEG and counters == NEG_QUOTA, f"neg 配额 {counters} ≠ {NEG_QUOTA}"
    return out, rejects


# ---------------------------------------------------------------- 主流程

def main() -> int:
    if VARIANT == "low":
        outs = ["aug-neg-low.jsonl", "swift-train-cn-low.jsonl", "aug-manifest-low.json"]
    else:
        outs = ["aug-garbled.jsonl", "aug-minimal.jsonl", "aug-neg.jsonl",
                "swift-train-cg.jsonl", "swift-train-cm.jsonl", "swift-train-cn.jsonl",
                "probe-negt.jsonl", "aug-manifest.json"]
    for p in [MEDC / f for f in outs]:
        assert not p.exists(), f"证据只增不改，已存在：{p}"
    src = load_f1()
    assert len(src) == 1154, f"f1 行数 {len(src)} ≠ 1154"
    sysp = system_prompt()
    eval_users = [json.loads(ln)["user"]
                  for f in ("eval-neg.jsonl", "eval-negn.jsonl")
                  for ln in (MEDC / f).read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(eval_users) == 30

    aug_n, _ = build_neg(src, sysp, eval_users)
    if VARIANT == "low":  # R5' 回退分支：只重造 cn 臂低配比（12 条），cg/cm 不重跑
        f1_rows = [json.loads(ln) for ln in F1.read_text(encoding="utf-8").splitlines() if ln.strip()]
        merged = f1_rows + aug_n
        (MEDC / "aug-neg-low.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in aug_n), encoding="utf-8")
        (MEDC / "swift-train-cn-low.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8")
        manifest = {
            "date": "2026-09-29", "variant": "low", "seed": SEED, "tool": "tools/augment_med_c.py",
            "reason": "R5' 回退分支预注册响应：合成配比下调重跑一次（36→12，A4/B6/C2，B 前 6 含冠军边界 miss B03/B04）",
            "source": {"file": "data/med0/swift-train-f1.jsonl", "n": len(src)},
            "n": len(aug_n), "quota": NEG_QUOTA, "merged": len(merged),
            "expected_train_steps": (len(merged) + 15) // 16 * 3,
            "note": "场景库前 N 配额确定性选样；值抽样流与 full 版独立（非子集关系）；断言门与 full 同规全跑",
        }
        (MEDC / "aug-manifest-low.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[augment-med-c low] cn 低配比 {len(aug_n)} 条 · 合并 {len(merged)} · 步数 {manifest['expected_train_steps']} · 断言全绿 ✓")
        return 0

    aug_g = build_garbled(src, sysp)
    aug_m, m2_stat = build_minimal(src, sysp)

    f1_rows = [json.loads(ln) for ln in F1.read_text(encoding="utf-8").splitlines() if ln.strip()]
    arms = {"cg": aug_g, "cm": aug_m, "cn": aug_n}
    steps = {}
    for tag, aug in arms.items():
        (MEDC / f"aug-{ {'cg': 'garbled', 'cm': 'minimal', 'cn': 'neg'}[tag] }.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in aug), encoding="utf-8")
        merged = f1_rows + aug  # LAB-E 臂合并约定：原序 + aug 追加
        (MEDC / f"swift-train-{tag}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8")
        steps[tag] = (len(merged) + 15) // 16 * 3  # bs1×accum16 · 3ep 末步预估（C2 硬门用）

    # neg 臂评测行视图（探针质量门用，评测器 EXTRA_SETS negt 键指此）
    probe_rows = [{"id": r["id"], "user": r["messages"][1]["content"],
                   "gold": {"diseases": ["无"],
                            "reasons": re.findall(r"(?m)^\s*\d+[\.、]\s*(.+)$", r["messages"][2]["content"])},
                   "split": "test_rob", "condition_kind": "negative", "expected_category": "无"}
                  for r in aug_n]
    (MEDC / "probe-negt.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in probe_rows), encoding="utf-8")

    manifest = {
        "date": "2026-09-29", "seed": SEED, "tool": "tools/augment_med_c.py",
        "source": {"file": "data/med0/swift-train-f1.jsonl", "n": len(src)},
        "leak_guard": "变体源全部 ∈ f1 train；neg 臂新生成且三层防线（主诉非逐字同 ∧ 全文 ratio<0.85 ∧ 叙事块 ratio<0.80）；评测集一字不入",
        "probe": "probe-negt.jsonl = aug-neg 的评测行视图（造数质量门：冠军反面探针，判病即修条；不入判卷）",
        "arms": {
            "cg": {"n": len(aug_g), "mode": "garbled 乱码注入", "quotes": "分条引用乱码字符（冠军形态）"},
            "cm": {"n": len(aug_m), "quota": {"m1": M1_QUOTA, "m2": M2_QUOTA},
                    "m2_tried": m2_stat["tried"],
                    "m2_deleted_fields_dist": {"/".join(t): sum(1 for r in aug_m if r["aug_meta"]["mode"] == "minimal-m2" and tuple(r["aug_meta"]["deleted"]) == t)
                                               for t in [("辅助检查",), ("体格检查",), ("辅助检查", "体格检查")]}},
            "cn": {"n": len(aug_n), "quota": NEG_QUOTA,
                    "anchors": "train 人口学（源性别年龄去重顺序取），病种见各条 aug_meta"},
        },
        "expected_train_steps": steps,
        "checks": ["fmt/语义门（score_item 复判）×108 全过", "幻觉教学侧零原子（aug 全臂）",
                    "garbled 全字符落字母表 ∧ 换行保持", "M2 长度≥60% ∧ 依据原子级过滤",
                    "neg 数值落 WS/T 锚 ∧ 现症阳性扫描零命中 ∧ 评测相似度<0.90",
                    "neg id 前缀 negt- 与评测 neg-/negn- 不交"],
    }
    (MEDC / "aug-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest["arms"], ensure_ascii=False, indent=1))
    print(f"[augment-med-c] 三臂 { {k: len(v) for k, v in arms.items()} } · 合并集 1190 ×3 · 断言全绿 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
