"""CN-GongWen-Audit（dataset_4）：公文审核/纠错任务。

给模型一份**可能含若干注入雷区**的公文，要求其依据《条例》与 GB/T 9704 找出全部问题
（完全合规则应明确"无问题"）。对标"审核清单 / 十项硬核自查"。

设计（延续仓库"零随机、逐字节复现"）：
- 复用 dataset_3 的确定性参考公文 deterministic_reference 作为"正确底稿"；
- 按 SHA-256 确定性**注入**一个违规子集（含约 1/5 完全合规的对照样本）；
- 金标准 = 注入的违规集合；校验器用独立**检测器**断言"注入即可检出"（金标准诚实）；
- 打分 `scorer.py --dataset audit`：违规类型的 precision/recall/F1 + 逐项精确匹配 + 合规样本零误报。
所有机关、字号、姓名均为合成示例。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collections import Counter

from benchmark_schema import DOC_TYPES, FORBIDDEN_PHRASES, MEDICAL_FORBIDDEN_PHRASES, doc_type_by_name
from generate_benchmarks import agency_metadata, hashed, write_jsonl
from generate_writing_prompts import (
    LENGTH_ORDER,
    WritingSpec,
    _make_spec,
    build_framework,
    build_writing_specs,
    deterministic_reference,
)

ROOT = SCRIPT_DIR.parent
DATASET4 = ROOT / "dataset_4_audit"

# 违规类型词表（代码 → 中文说明），写进 prompt 供模型选择，确定性可打分
VIOLATION_TYPES: tuple[tuple[str, str], ...] = (
    ("title_missing_doctype", "标题缺少文种（标题三要素不全）"),
    ("year_square_bracket", "发文字号年份误用方括号（应为六角括号〔〕）"),
    ("seq_add_di", "发文字号顺序号误加“第”字"),
    ("date_chinese", "成文日期误用汉字数字（应为阿拉伯数字）"),
    ("ordinal_dunhao", "第二层层次序号“（一）”后误加顿号"),
    ("hype_language", "使用夸大/网络化/绝对化等不规范表述"),
    ("fabricated_legal_article", "编造法规条款依据（如“第99条”）"),
    ("qingshi_multihead", "请示多头主送（违反一文一事、单一主送）"),
    ("missing_signatory_upward", "上行文缺少签发人"),
    ("baogao_embeds_request", "报告夹带请示事项"),
    ("cc_to_subordinate_upward", "上行文抄送下级机关"),
    # —— 医疗卫生专属违规 ——
    ("overclaim_cure", "夸大/保证疗效（治愈、根治、无副作用、疗效领先等）"),
    ("patient_privacy_leak", "泄露可识别患者隐私（姓名、住址、病历截图等）"),
    ("research_as_clinical", "把科研探索写成临床常规/全面推广，未经验证或伦理审批"),
    ("ai_replaces_physician", "AI/系统替代医师独立诊疗（自动诊断、无需医生审核）"),
    ("medical_insurance_fraud", "医保基金违规风险表述（分解住院、串换项目、诱导住院等）"),
)
VIOLATION_CODES = tuple(code for code, _ in VIOLATION_TYPES)

_CN_DIGIT = {"0": "〇", "1": "一", "2": "二", "3": "三", "4": "四",
             "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}


def _to_cn(num: str) -> str:
    return "".join(_CN_DIGIT[c] for c in str(num))


# --------------------------------------------------------------------------------------
# 注入器：在“正确底稿”上制造可检出的违规（lines 原地修改）
# --------------------------------------------------------------------------------------
def _title_idx(lines: list[str]) -> int:
    return next(i for i, l in enumerate(lines) if "关于" in l)


def inj_title_missing(lines: list[str], spec: WritingSpec) -> None:
    i = _title_idx(lines)
    suffix = "的" + spec.doc_type
    if lines[i].endswith(suffix):
        lines[i] = lines[i][: -len(suffix)]


def inj_year_square(lines: list[str], spec: WritingSpec) -> None:
    for i, l in enumerate(lines):
        if "〔" in l and "号" in l:
            lines[i] = l.replace("〔", "[").replace("〕", "]")
            return


def inj_seq_di(lines: list[str], spec: WritingSpec) -> None:
    for i, l in enumerate(lines):
        if ("〔" in l or "[" in l) and "号" in l:
            lines[i] = re.sub(r"(\d+)号", r"第\1号", l, count=1)
            return


def inj_date_chinese(lines: list[str], spec: WritingSpec) -> None:
    for i in range(len(lines) - 1, -1, -1):  # 末尾的成文日期（落款）
        s = lines[i].strip("　 ")
        m = re.fullmatch(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", s)
        if m:
            y, mo, d = m.groups()
            lines[i] = lines[i].replace(s, f"{_to_cn(y)}年{_to_cn(mo)}月{_to_cn(d)}日")
            return


def inj_ordinal_dunhao(lines: list[str], spec: WritingSpec) -> None:
    for i, l in enumerate(lines):
        if "（一）" in l:
            lines[i] = l.replace("（一）", "（一）、", 1)
            return


def inj_hype(lines: list[str], spec: WritingSpec) -> None:
    for i, l in enumerate(lines):
        if "如下：" in l:
            lines[i] = l + "本项工作全方位赋能、绝对领先，将完美闭环、全面解决所有问题。"
            return


def inj_fake_article(lines: list[str], spec: WritingSpec) -> None:
    for i, l in enumerate(lines):
        if "如下：" in l:
            lines[i] = l.replace("如下：", "（依据《公文管理办法》第99条）如下：", 1)
            return


def inj_qingshi_multihead(lines: list[str], spec: WritingSpec) -> None:
    for i, l in enumerate(lines):
        if l.endswith("：") and spec.recipient and spec.recipient in l:
            lines[i] = "示范市人民政府、示范市财政局、示范市发展和改革委员会："
            return


def inj_signatory_removed(lines: list[str], spec: WritingSpec) -> None:
    for i, l in enumerate(lines):
        if l.startswith("签发人"):
            lines.pop(i)
            return


def inj_baogao_request(lines: list[str], spec: WritingSpec) -> None:
    for i, l in enumerate(lines):
        if "特此报告" in l:
            lines.insert(i, "　　请予批准上述事项。")
            return


def inj_cc_subordinate_upward(lines: list[str], spec: WritingSpec) -> None:
    lines.append("抄送：各县（市、区）有关单位。")


def _insert_after_lead(lines: list[str], sentence: str) -> None:
    for i, l in enumerate(lines):
        if "如下：" in l:
            lines.insert(i + 1, sentence)
            return
    lines.insert(min(1, len(lines)), sentence)


def inj_overclaim_cure(lines: list[str], spec: WritingSpec) -> None:
    _insert_after_lead(lines, "　　本疗法确保治愈、无副作用，治愈率100%、疗效领先。")


def inj_patient_privacy(lines: list[str], spec: WritingSpec) -> None:
    _insert_after_lead(lines, "　　患者张某，男，47岁，示范区幸福小区人，现附完整病历截图。")


def inj_research_as_clinical(lines: list[str], spec: WritingSpec) -> None:
    _insert_after_lead(lines, "　　本技术已证明可全面推广、替代传统治疗，无需进一步验证。")


def inj_ai_replaces(lines: list[str], spec: WritingSpec) -> None:
    _insert_after_lead(lines, "　　智能系统自动诊断并直接生成处方，无需医生审核。")


def inj_insurance_fraud(lines: list[str], spec: WritingSpec) -> None:
    _insert_after_lead(lines, "　　可通过分解住院、串换项目提高医保结算金额。")


# code -> (applicable(spec), apply(lines, spec))
INJECTORS: dict[str, tuple[Callable[[WritingSpec], bool], Callable[[list[str], WritingSpec], None]]] = {
    "title_missing_doctype": (lambda s: True, inj_title_missing),
    "seq_add_di": (lambda s: True, inj_seq_di),
    "year_square_bracket": (lambda s: True, inj_year_square),
    "date_chinese": (lambda s: True, inj_date_chinese),
    "ordinal_dunhao": (lambda s: s.length != "short", inj_ordinal_dunhao),
    "hype_language": (lambda s: True, inj_hype),
    "fabricated_legal_article": (lambda s: True, inj_fake_article),
    "qingshi_multihead": (lambda s: s.doc_type == "请示", inj_qingshi_multihead),
    "missing_signatory_upward": (lambda s: s.needs_signatory, inj_signatory_removed),
    "baogao_embeds_request": (lambda s: s.doc_type == "报告", inj_baogao_request),
    "cc_to_subordinate_upward": (lambda s: s.direction == "upward", inj_cc_subordinate_upward),
    "overclaim_cure": (lambda s: s.policy_domain == "医疗卫生", inj_overclaim_cure),
    "patient_privacy_leak": (lambda s: s.policy_domain == "医疗卫生", inj_patient_privacy),
    "research_as_clinical": (lambda s: s.policy_domain == "医疗卫生", inj_research_as_clinical),
    "ai_replaces_physician": (lambda s: s.policy_domain == "医疗卫生", inj_ai_replaces),
    "medical_insurance_fraud": (lambda s: s.policy_domain == "医疗卫生", inj_insurance_fraud),
}
# 固定施加顺序，保证字号类注入可叠加（先加“第”再换方括号）
CANONICAL_ORDER = (
    "title_missing_doctype", "seq_add_di", "year_square_bracket", "date_chinese",
    "ordinal_dunhao", "hype_language", "fabricated_legal_article", "qingshi_multihead",
    "missing_signatory_upward", "baogao_embeds_request", "cc_to_subordinate_upward",
    "overclaim_cure", "patient_privacy_leak", "research_as_clinical",
    "ai_replaces_physician", "medical_insurance_fraud",
)


def detect_violations(text: str, spec: WritingSpec) -> set[str]:
    """独立检测器：从公文文本检出违规类型（用于校验金标准诚实性）。"""
    found: set[str] = set()
    lines = text.split("\n")
    title = next((l for l in lines if "关于" in l), "")
    if title and not title.rstrip().endswith(spec.doc_type):
        found.add("title_missing_doctype")
    if re.search(r"\[20\d{2}\]", text):
        found.add("year_square_bracket")
    if re.search(r"[〕\]]第\d+号", text):
        found.add("seq_add_di")
    if re.search(r"[〇一二三四五六七八九]{2,}年[一二三四五六七八九十]+月", text):
        found.add("date_chinese")
    if "（一）、" in text:
        found.add("ordinal_dunhao")
    if any(p in text for p in FORBIDDEN_PHRASES):
        found.add("hype_language")
    if re.search(r"第\d+条", text):
        found.add("fabricated_legal_article")
    if spec.doc_type == "请示":
        rl = next((l for l in lines if l.endswith("：")), "")
        if "、" in rl:
            found.add("qingshi_multihead")
    if spec.needs_signatory and "签发人" not in text:
        found.add("missing_signatory_upward")
    if spec.doc_type == "报告" and "请予批准" in text:
        found.add("baogao_embeds_request")
    if spec.direction == "upward" and "抄送：" in text:
        found.add("cc_to_subordinate_upward")
    # 医疗卫生专属违规
    if any(p in text for p in ("确保治愈", "治愈率100%", "疗效领先", "无副作用", "根治", "包治", "保证治好", "完全避免并发症", "一次见效")):
        found.add("overclaim_cure")
    if "病历截图" in text or "小区人" in text or re.search(r"患者[张李王刘赵]某", text):
        found.add("patient_privacy_leak")
    if "全面推广" in text and "替代" in text:
        found.add("research_as_clinical")
    if "无需医生审核" in text:
        found.add("ai_replaces_physician")
    if "分解住院" in text or "串换项目" in text:
        found.add("medical_insurance_fraud")
    return found


# 每个违规类型的目标最小样本量（"充裕"门槛）。未显式指定时按题量缩放
# （standard=90 → 10，full=180 → 20），可用 CN_GW_AUDIT_FLOOR 覆盖。
_AUDIT_FLOOR_ENV = os.getenv("CN_GW_AUDIT_FLOOR")
AUDIT_MAX_PER_DOC = int(os.getenv("CN_GW_AUDIT_MAX_PER_DOC", "5"))


def _default_floor(count: int) -> int:
    return int(_AUDIT_FLOOR_ENV) if _AUDIT_FLOOR_ENV else max(8, round(count / 9))
# 文种专属违规的承载文种较稀缺，需在审核集中适当多采，否则其样本量天然偏低。
_SCARCE_DOCTYPES = {"请示", "报告", "议案"}


def build_audit_specs(count: int) -> list[WritingSpec]:
    """构造审核集的 WritingSpec 列表，按"承载力"加权分配文种。

    通用违规几乎所有文种都能承载，样本量不愁；但 ``qingshi_multihead`` 只在请示、
    ``baogao_embeds_request`` 只在报告、``missing_signatory_upward``/
    ``cc_to_subordinate_upward`` 只在上行文（报告/请示/议案）。复用写作集的均匀分布
    （每文种 6 篇）会让这些类型天然样本不足。这里给请示/报告/议案更高配额，并保证
    15 文种全覆盖、约一半医疗，使每个违规类型都有足够承载篇目达到 ``AUDIT_FLOOR``。
    """
    agencies = agency_metadata(37)
    by_code = {a["agency_code"]: a for a in agencies}
    names = [d.name for d in DOC_TYPES]
    # 承载力加权：请示/报告各 12 权重、议案 6、其余各 5（count=90 时精确 12/12/6/5×12）。
    weights = {n: 5 for n in names}
    weights["请示"] = 12
    weights["报告"] = 12
    weights["议案"] = 6
    total_w = sum(weights.values())
    raw = {n: count * weights[n] / total_w for n in names}
    quota = {n: int(raw[n]) for n in names}
    # 余数按小数部分从大到小补齐（确定性），保证 sum(quota)==count。
    remainder = count - sum(quota.values())
    for n in sorted(names, key=lambda n: (-(raw[n] - int(raw[n])), n))[:max(0, remainder)]:
        quota[n] += 1

    plan_seq: list[tuple[Any, str, bool]] = []
    for n in names:
        dt = doc_type_by_name(n)
        for j in range(quota[n]):
            length = LENGTH_ORDER[j % len(LENGTH_ORDER)]
            medical = (j % 2 == 0)  # 约一半医疗
            plan_seq.append((dt, length, medical))
    return [
        _make_spec(dt, length, medical, i, agencies, by_code)
        for i, (dt, length, medical) in enumerate(plan_seq)
    ]


def _build_flawed(spec: WritingSpec, correct: str, codes: set[str]) -> tuple[str, set[str]]:
    """按 CANONICAL_ORDER 注入 codes，返回 (含缺陷公文, 独立检测器实际检出的违规集合)。

    金标准取**检测器实际检出**而非计划注入集，故 honesty（gold==detect(flawed)）由构造
    保证；某注入器在该篇无效（no-op）时自然不计入，门槛由迭代补足。
    """
    lines = correct.split("\n")
    for code in CANONICAL_ORDER:
        if code in codes:
            INJECTORS[code][1](lines, spec)
    flawed = "\n".join(lines)
    return flawed, detect_violations(flawed, spec)


def build_audit_dataset(count: int, floor: int | None = None):
    if floor is None:
        floor = _default_floor(count)
    specs = build_audit_specs(count)
    n = len(specs)
    refs = [deterministic_reference(s, build_framework(s)) for s in specs]

    # 合规对照（约 1/5）：只从"非稀缺承载"文种抽取，避免饿死稀缺类型的门槛。
    clean_target = max(1, n // 5)
    clean_pool = sorted(
        (i for i in range(n) if specs[i].doc_type not in _SCARCE_DOCTYPES),
        key=lambda i: hashed("auclean", specs[i].spec_id),
    )
    is_clean = set(clean_pool[:clean_target])

    plan: dict[int, set[str]] = {i: set() for i in range(n)}
    tried: dict[int, set[str]] = {i: set() for i in range(n)}  # 在该篇 no-op 的 code

    def applicable(i: int, code: str) -> bool:
        return INJECTORS[code][0](specs[i])

    detected: dict[int, set[str]] = {i: set() for i in range(n)}
    for _ in range(8):  # 迭代补足门槛；收敛即停
        for i in range(n):
            if i in is_clean:
                detected[i] = set()
                continue
            _, det = _build_flawed(specs[i], refs[i], plan[i])
            detected[i] = det
            for c in list(plan[i]):  # 丢弃计划注入但未被检出的 no-op，记入 tried
                if c not in det:
                    plan[i].discard(c)
                    tried[i].add(c)
        counts = Counter(c for d in detected.values() for c in d)
        deficits = [(c, floor - counts[c]) for c in VIOLATION_CODES if counts[c] < floor]
        if not deficits:
            break
        progressed = False
        for code, need in deficits:
            cands = [
                i for i in range(n)
                if i not in is_clean and code not in detected[i]
                and code not in tried[i] and applicable(i, code)
                and len(plan[i]) < AUDIT_MAX_PER_DOC
            ]
            cands.sort(key=lambda i: (len(plan[i]), hashed("aufloor", code, specs[i].spec_id)))
            for i in cands[:need]:
                plan[i].add(code)
                progressed = True
        if not progressed:
            break

    public, hidden = [], []
    for i, s in enumerate(specs):
        correct = refs[i]
        if i in is_clean:
            flawed, det = correct, set()
        else:
            flawed, det = _build_flawed(s, correct, plan[i])
        violations = sorted(det, key=VIOLATION_CODES.index)
        qid = s.spec_id.replace("WP_", "AU_")
        audit_prompt = AUDIT_INSTRUCTION + flawed
        rewrite_prompt = REWRITE_INSTRUCTION + flawed
        public.append({"question_id": qid, "prompt": audit_prompt, "rewrite_prompt": rewrite_prompt})
        hidden.append({
            "question_id": qid, "prompt": audit_prompt, "rewrite_prompt": rewrite_prompt,
            "spec": asdict(s), "doc_type": s.doc_type, "length_bucket": s.length,
            "flawed_document": flawed, "violations": violations, "violation_count": len(violations),
            "is_clean": not violations, "corrected_document": correct,
        })
    return public, hidden


_VOCAB = "\n".join(f"- {code}：{desc}" for code, desc in VIOLATION_TYPES)
AUDIT_INSTRUCTION = (
    "请依据《党政机关公文处理工作条例》与 GB/T 9704—2012 审核下面这份公文，"
    "找出其中存在的所有问题；如果完全合规，请明确回答“无问题”。\n"
    "请从下列问题类型中选出存在的项（可多选，也可为空），输出其代码列表：\n"
    f"{_VOCAB}\n\n【待审公文】\n"
)


REWRITE_INSTRUCTION = (
    "下面这份公文可能存在若干不规范之处。请依据《党政机关公文处理工作条例》与 GB/T 9704—2012"
    "（医疗卫生题还须遵循相关医疗法规与伦理、隐私、医保要求），将其改写为一份合规公文："
    "纠正全部问题，保留文种、发文机关与事由等关键信息，规范标题三要素、层次序数、署名与成文日期。"
    "仅输出改写后的公文全文。\n\n【原公文】\n"
)


def build_audit_taxonomy() -> dict[str, Any]:
    return {
        "description": "CN-GongWen-Audit：公文审核/纠错（注入雷区→找出违规）",
        "violation_types": [{"code": c, "desc": d} for c, d in VIOLATION_TYPES],
        "audit_metrics": ["violation_precision", "violation_recall", "violation_f1",
                          "exact_match_rate", "clean_doc_accuracy"],
        "rewrite_metrics": ["violations_removed_rate", "facts_preserved_rate",
                            "format_valid_rate", "overall_rewrite_compliance"],
    }


def write_audit_dataset(count: int) -> dict[str, Any]:
    public, hidden = build_audit_dataset(count)
    write_jsonl(DATASET4 / "audit_tasks_public.jsonl", public)
    write_jsonl(DATASET4 / "audit_tasks_with_gold.jsonl", hidden)
    (DATASET4 / "taxonomy.json").write_text(
        json.dumps(build_audit_taxonomy(), ensure_ascii=False, indent=2), encoding="utf-8")
    clean = sum(1 for h in hidden if h["is_clean"])
    covered = {v for h in hidden for v in h["violations"]}
    return {
        "audit_count": len(public),
        "audit_clean": clean,
        "audit_flawed": len(public) - clean,
        "audit_violation_coverage": len(covered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成公文审核/纠错任务（注入雷区→找出违规）")
    parser.add_argument("--count", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(write_audit_dataset(args.count), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
