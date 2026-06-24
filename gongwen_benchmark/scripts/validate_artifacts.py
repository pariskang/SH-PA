"""Strict cross-file validation for generated CN official-document artifacts."""
from __future__ import annotations
import argparse, csv, json, re, sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from tokens import estimate_tokens

AGENCY_ID_PATTERN = re.compile(r"GA\d{3}")

# 与 generate_benchmarks.SUFFIXES 保持一致（用于统计 padding 占比）
SUFFIXES = (
    "请说明所依据的条例或国标条款",
    "请按行文规范程度逐项说明",
    "只依据《党政机关公文处理工作条例》和GB/T 9704回答",
    "如信息不足请先说明需要补充的要素",
)

REQUIRED_TASK_TYPES = {
    "direct_lookup", "cross_agency_ranking", "period_comparison", "sustained_trend",
    "composite_element_explanation", "anomaly_detection", "priority_ranking", "briefing",
    "cross_doc_extremum", "consecutive_compliance_streak", "counterfactual_format",
    "quality_filtered_aggregate", "negative_enumeration", "multi_criteria_ranking",
    "precision_percentage_change", "policy_domain_classification",
}
REQUIRED_BRIEFING_SUBTYPES = {"risk_focused_targeted", "conflicting_signals_briefing", "exclusion_briefing"}
REQUIRED_Q_TYPES = {
    "single_doc_type", "multi_doc_type", "cross_element_chain", "temporal_compound",
    "conflicting_signals", "boundary_precision", "negative_enumeration",
    "management_open", "ambiguous_boundary", "hallucination_trap", "spoken_noisy",
    "doctype_misuse", "addressing_relation", "authority_boundary", "medical_compliance",
}


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(root: Path) -> dict[str, Any]:
    d1 = root / "dataset_1_question_only"
    d2 = root / "dataset_2_data_qa"
    public = jsonl(d1 / "questions_public.jsonl")
    hidden = jsonl(d1 / "questions_with_hidden_metadata.jsonl")
    questions = jsonl(d2 / "questions.jsonl")
    answers = jsonl(d2 / "answers.jsonl")
    maps = jsonl(d2 / "evidence_map.jsonl")
    anomalies = jsonl(d2 / "anomaly_labels.jsonl")
    with (d2 / "records.csv").open(encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    # --- Q 公开/隐藏切分一致性与隔离 ---
    assert all(set(row) == {"question"} for row in public), "public split must contain question only"
    assert len(public) == len(hidden) and [x["question"] for x in public] == [x["question"] for x in hidden]
    assert len({x["question_id"] for x in hidden}) == len(hidden), "duplicate Q question_id"

    # --- DataQA 跨文件 ID 一致性 ---
    qids = {x["question_id"] for x in questions}
    assert qids == {x["question_id"] for x in answers} == {x["question_id"] for x in maps}, "DataQA id mismatch"
    doc_ids = {x["doc_id"] for x in records}
    assert len(doc_ids) == len(records), "duplicate doc_id in records.csv"
    for row in questions + answers + maps:
        assert set(row.get("evidence_rows", [])) <= doc_ids, "evidence_rows reference unknown doc_id"
    assert all(x["doc_id"] in doc_ids for x in anomalies), "anomaly label references unknown doc_id"
    assert all(a["calculation"] and a["confidence"] in {"high", "medium", "low"} for a in answers), "answer contract"

    # --- DataQA 任务类型覆盖（8 基线 + 7 进阶共 15 种）---
    task_types = Counter(x["task_type"] for x in questions)
    missing = REQUIRED_TASK_TYPES - set(task_types)
    assert not missing, f"missing task types: {missing}"

    # --- 播报子类型至少包含 3 种高级格式 ---
    briefing_subtypes = Counter(x.get("briefing_subtype", "unspecified") for x in questions if x["task_type"] == "briefing")
    missing_briefings = REQUIRED_BRIEFING_SUBTYPES - set(briefing_subtypes)
    assert not missing_briefings, f"missing briefing subtypes: {missing_briefings}"

    # --- Q 问题类型覆盖（11 类）---
    q_types = Counter(x["question_type"] for x in hidden)
    missing_q = REQUIRED_Q_TYPES - set(q_types)
    assert not missing_q, f"missing Q question types: {missing_q}"

    # --- Q 难度须偏向高难（hard ≥40%，easy ≤25%）---
    diff = Counter(x["difficulty"] for x in hidden)
    total = len(hidden)
    hard_share = diff.get("hard", 0) / total if total else 0
    easy_share = diff.get("easy", 0) / total if total else 0
    assert hard_share >= 0.40, f"hard share {hard_share:.2%} below 40% — benchmark too easy"
    assert easy_share <= 0.25, f"easy share {easy_share:.2%} above 25% — too many trivial questions"

    # --- 幻觉陷阱须多样化（≥12 种不同 trap_type）---
    trap_types = {x["expected_slots"].get("trap_type") for x in hidden if x["question_type"] == "hallucination_trap"}
    trap_types.discard(None)
    assert len(trap_types) >= 12, f"hallucination traps too repetitive: {len(trap_types)} distinct, need ≥12"

    # --- 播报事实接地：final_answer 中引用的 GA### 必须出现在该题证据机关集合内 ---
    by_doc = {x["doc_id"]: x for x in records}
    answer_by_qid = {a["question_id"]: a for a in answers}
    ungrounded = []
    for q in questions:
        if q["task_type"] != "briefing":
            continue
        a = answer_by_qid.get(q["question_id"])
        if not a:
            continue
        ev_agencies = {by_doc[r]["agency_id"] for r in q.get("evidence_rows", []) if r in by_doc}
        cited = set(AGENCY_ID_PATTERN.findall(a["final_answer"]))
        if cited - ev_agencies:
            ungrounded.append((q["question_id"], sorted(cited - ev_agencies)))
    assert not ungrounded, f"briefings cite non-evidence agencies: {ungrounded[:3]}"

    # --- padding 占比：尾缀变体 ≤40%，保证独立推理内容占主导 ---
    padded = sum(1 for h in hidden if any(s in h["question"] for s in SUFFIXES))
    pad_share = padded / len(hidden) if hidden else 0
    assert pad_share <= 0.40, f"padding share {pad_share:.2%} above 40% — too many suffix variants"

    # --- 医疗政策方向须约占一半，且覆盖足够多的医疗子领域 ---
    corpus_medical = sum(1 for r in records if r.get("policy_domain") == "医疗卫生")
    corpus_medical_share = corpus_medical / len(records) if records else 0
    medical_areas = {r.get("medical_area") for r in records if r.get("medical_area")}
    q_medical = sum(1 for h in hidden if h.get("policy_domain") == "医疗卫生")
    q_medical_share = q_medical / len(hidden) if hidden else 0
    # 宽松合格门槛（跨 profile 稳健）；提交的 standard 档由单元测试施加 45%~55% 的严格约束
    assert 0.40 <= corpus_medical_share <= 0.60, f"corpus medical share {corpus_medical_share:.2%} not ~half"
    assert 0.40 <= q_medical_share <= 0.60, f"Q medical share {q_medical_share:.2%} not ~half"
    assert len(medical_areas) >= 10, f"medical areas covered {len(medical_areas)} < 10 — insufficient breadth"
    # DataQA 分类任务须同时覆盖医疗与通用两类公文
    cls_qids = {q["question_id"] for q in questions if q["task_type"] == "policy_domain_classification"}
    classify = [a for a in answers if a["question_id"] in cls_qids]
    cls_medical = sum(1 for a in classify if isinstance(a.get("answer_value"), dict) and a["answer_value"].get("policy_domain") == "医疗卫生")
    assert classify, "policy_domain_classification task missing"
    assert 0 < cls_medical < len(classify), "classification task must include both medical and general docs"

    # --- 医疗子领域须进一步细化为足够多的具体政策主题，且医疗分类答案须带具体主题 ---
    medical_topics = {r.get("medical_topic") for r in records if r.get("medical_topic")}
    assert len(medical_topics) >= 40, f"medical topics covered {len(medical_topics)} < 40 — insufficient granularity"
    cls_medical_with_topic = sum(
        1 for a in classify
        if isinstance(a.get("answer_value"), dict)
        and a["answer_value"].get("policy_domain") == "医疗卫生"
        and a["answer_value"].get("medical_topic")
    )
    assert cls_medical_with_topic == cls_medical, "every medical classification answer must carry a specific topic"

    return {
        "q": len(public),
        "dataqa": len(questions),
        "records": len(records),
        "anomalies": len(anomalies),
        "task_types": dict(task_types),
        "briefing_subtypes": dict(briefing_subtypes),
        "q_question_types": dict(q_types),
        "q_difficulty": dict(diff),
        "q_hard_share": round(hard_share, 3),
        "q_trap_diversity": len(trap_types),
        "q_padding_share": round(pad_share, 3),
        "briefing_ungrounded_count": len(ungrounded),
        "corpus_medical_share": round(corpus_medical_share, 3),
        "q_medical_share": round(q_medical_share, 3),
        "medical_area_coverage": len(medical_areas),
        "medical_topic_coverage": len(medical_topics),
        "classification_medical": cls_medical,
        "classification_total": len(classify),
        **validate_writing(root),
        **validate_audit(root),
    }


def validate_writing(root: Path) -> dict[str, Any]:
    """CN-GongWen-Writing（dataset_3）：公文写作测试 prompt 的跨文件校验。"""
    d3 = root / "dataset_3_writing"
    if not (d3 / "writing_prompts_public.jsonl").exists():
        return {}  # 兼容尚未生成 dataset_3 的旧工件
    public = jsonl(d3 / "writing_prompts_public.jsonl")
    hidden = jsonl(d3 / "writing_prompts_with_rubric.jsonl")

    # --- public/hidden 隔离与 ID 一致 ---
    assert all(set(p) <= {"question_id", "prompt"} for p in public), "writing public must hold only question_id+prompt"
    assert all("reference_answer" not in p and "rubric" not in p for p in public), "writing public leaks gold"
    assert {p["question_id"] for p in public} == {h["question_id"] for h in hidden}, "writing id mismatch"
    assert len({h["question_id"] for h in hidden}) == len(hidden), "duplicate writing question_id"

    # --- 三种长度分桶齐备 ---
    buckets = Counter(h["length_bucket"] for h in hidden)
    assert set(buckets) == {"short", "medium", "long"}, f"writing length buckets incomplete: {set(buckets)}"

    # --- 文种覆盖与合法行文方向、rubric 完整性 ---
    doc_types = {h["spec"]["doc_type"] for h in hidden}
    assert len(doc_types) >= 10, f"writing doc-type coverage {len(doc_types)} < 10"
    for h in hidden:
        r = h["rubric"]
        assert r["direction"] in {"upward", "downward", "parallel"}, "illegal writing direction"
        lo, hi = r["target_tokens"]
        assert 0 < lo < hi, "bad writing target_tokens"
        assert r["required_elements"] and r["min_sections"] >= 2, "incomplete writing rubric"

    # --- 参考公文须落入目标 token 区间（确定性真相自洽）---
    in_range = sum(1 for h in hidden
                   if h["rubric"]["target_tokens"][0] <= estimate_tokens(h["reference_answer"]) <= h["rubric"]["target_tokens"][1])
    assert in_range == len(hidden), f"writing references out of token range: {len(hidden) - in_range}"

    # --- prompt 须忠实包含文种与目标 token 下限（事实接地）---
    grounded = sum(1 for h in hidden
                   if h["spec"]["doc_type"] in h["prompt"] and str(h["rubric"]["target_tokens"][0]) in h["prompt"])
    assert grounded == len(hidden), f"writing prompts not fact-grounded: {len(hidden) - grounded}"

    medical = sum(1 for h in hidden if h["spec"]["policy_domain"] == "医疗卫生")
    return {
        "writing": len(public),
        "writing_buckets": dict(buckets),
        "writing_doc_type_coverage": len(doc_types),
        "writing_reference_in_range_share": round(in_range / len(hidden), 3),
        "writing_prompt_grounded_share": round(grounded / len(hidden), 3),
        "writing_medical_share": round(medical / len(hidden), 3),
    }


def validate_audit(root: Path) -> dict[str, Any]:
    """CN-GongWen-Audit（dataset_4）：审核/纠错任务的跨文件校验与金标准诚实性检查。"""
    d4 = root / "dataset_4_audit"
    if not (d4 / "audit_tasks_public.jsonl").exists():
        return {}  # 兼容尚未生成 dataset_4 的旧工件
    public = jsonl(d4 / "audit_tasks_public.jsonl")
    hidden = jsonl(d4 / "audit_tasks_with_gold.jsonl")

    assert all(set(p) <= {"question_id", "prompt", "rewrite_prompt"} for p in public), "audit public keys unexpected"
    assert all("violations" not in p and "corrected_document" not in p for p in public), "audit public leaks gold"
    assert {p["question_id"] for p in public} == {h["question_id"] for h in hidden}, "audit id mismatch"
    assert len({h["question_id"] for h in hidden}) == len(hidden), "duplicate audit question_id"

    # 金标准诚实：独立检测器须与注入的违规集合逐项一致；纠错改写金标准（corrected_document）须本身合规
    from generate_audit_tasks import VIOLATION_CODES, detect_violations
    from generate_writing_prompts import build_writing_specs
    specs = {s.spec_id.replace("WP_", "AU_"): s for s in build_writing_specs(len(hidden))}
    codes = set(VIOLATION_CODES)
    honest = corrected_clean = 0
    for h in hidden:
        assert set(h["violations"]) <= codes, "unknown audit violation code"
        spec = specs[h["question_id"]]
        if detect_violations(h["flawed_document"], spec) == set(h["violations"]):
            honest += 1
        if not detect_violations(h["corrected_document"], spec):
            corrected_clean += 1
    assert honest == len(hidden), f"audit gold not detector-consistent: {len(hidden) - honest}"
    assert corrected_clean == len(hidden), f"audit corrected_document not clean: {len(hidden) - corrected_clean}"

    clean = sum(1 for h in hidden if h["is_clean"])
    assert clean > 0 and (len(hidden) - clean) > 0, "audit needs both clean and flawed docs"
    covered = {v for h in hidden for v in h["violations"]}
    return {
        "audit": len(public),
        "audit_clean": clean,
        "audit_flawed": len(hidden) - clean,
        "audit_violation_coverage": len(covered),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    print(json.dumps(validate(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
