"""Strict cross-file validation for generated CN official-document artifacts."""
from __future__ import annotations
import argparse, csv, json, re
from collections import Counter
from pathlib import Path
from typing import Any

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
    "precision_percentage_change",
}
REQUIRED_BRIEFING_SUBTYPES = {"risk_focused_targeted", "conflicting_signals_briefing", "exclusion_briefing"}
REQUIRED_Q_TYPES = {
    "single_doc_type", "multi_doc_type", "cross_element_chain", "temporal_compound",
    "conflicting_signals", "boundary_precision", "negative_enumeration",
    "management_open", "ambiguous_boundary", "hallucination_trap", "spoken_noisy",
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
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    print(json.dumps(validate(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
