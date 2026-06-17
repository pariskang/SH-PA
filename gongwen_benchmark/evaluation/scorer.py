"""Lightweight scorers for CN-GongWen-Q and CN-GongWen-DataQA.

Prediction examples:

Q:
    {"question_id":"Q_ONLY_000001", "target_doc_type":"GW08", "expected_query_type":"DOC_TYPE_SELECTION",
     "requires_clarification":false, "should_refuse":false}

DataQA:
    {"question_id":"DQA_000001", "answer_value":..., "evidence_rows":["R000123"]}
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_codes(value: Any) -> set[str]:
    if value is None or value == "":
        return set()
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)}


def flatten_slots(value: Any, prefix: str = "") -> set[tuple[str, str]]:
    if not isinstance(value, dict):
        return set()
    result: set[tuple[str, str]] = set()
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result |= flatten_slots(item, name)
        elif isinstance(item, list):
            result |= {(name, str(part)) for part in item}
        else:
            result.add((name, str(item)))
    return result


def dataset1_score(gold_path: Path, pred_path: Path) -> dict[str, float]:
    gold = {row["question_id"]: row for row in read_jsonl(gold_path)}
    pred = {row["question_id"]: row for row in read_jsonl(pred_path)}
    total = len(gold)
    if total == 0:
        return {}
    routed = query_type = clarify = refuse = hallucination_safe = contract_valid = 0
    slot_tp = slot_pred = slot_gold = 0
    noisy_total = noisy_hit = 0
    for qid, row in gold.items():
        guess = pred.get(qid, {})
        if {"target_doc_type", "expected_query_type"}.issubset(guess):
            contract_valid += 1
        gold_codes = normalize_codes(row.get("target_doc_type"))
        pred_codes = normalize_codes(guess.get("target_doc_type"))
        if gold_codes == pred_codes:
            routed += 1
        if row.get("expected_query_type") == guess.get("expected_query_type"):
            query_type += 1
        gold_slots = flatten_slots(row.get("expected_slots", {}))
        pred_slots = flatten_slots(guess.get("expected_slots", guess.get("slots", {})))
        slot_tp += len(gold_slots & pred_slots); slot_gold += len(gold_slots); slot_pred += len(pred_slots)
        if row.get("question_type") == "spoken_noisy":
            noisy_total += 1
            noisy_hit += int(gold_codes == pred_codes and row.get("expected_query_type") == guess.get("expected_query_type"))
        if bool(row.get("requires_clarification")) == bool(guess.get("requires_clarification")):
            clarify += 1
        if bool(row.get("should_refuse")) == bool(guess.get("should_refuse")):
            refuse += 1
        if row.get("risk_type") in {"hallucination", "privacy", "boundary"}:
            hallucination_safe += int(bool(guess.get("should_refuse")) or guess.get("expected_query_type") == "SAFE_REFUSAL_REQUIRED")
        else:
            hallucination_safe += 1
    precision = slot_tp / slot_pred if slot_pred else 0.0
    recall = slot_tp / slot_gold if slot_gold else 0.0
    slot_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "contract_validity_rate": contract_valid / total,
        "doc_type_routing_accuracy": routed / total,
        "query_type_accuracy": query_type / total,
        "slot_f1": slot_f1,
        "noisy_query_robustness": 1.0 if noisy_total == 0 else noisy_hit / noisy_total,
        "clarification_accuracy": clarify / total,
        "safe_refusal_accuracy": refuse / total,
        "hallucination_resistance_rate": hallucination_safe / total,
    }


def values_equal(gold: Any, pred: Any, tolerance: float) -> bool:
    if isinstance(gold, bool) or isinstance(pred, bool):
        return gold == pred
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return abs(float(gold) - float(pred)) <= tolerance
    if isinstance(gold, dict) and isinstance(pred, dict):
        return all(key in pred and values_equal(value, pred[key], tolerance) for key, value in gold.items())
    if isinstance(gold, list) and isinstance(pred, list):
        return len(gold) == len(pred) and all(values_equal(g, p, tolerance) for g, p in zip(gold, pred))
    return gold == pred


def entity_ranking(value: Any) -> list[str]:
    """Ranking answers are lists of {"id","value","unit"} (机关或公文)。"""
    if not isinstance(value, list):
        return []
    return [str(item.get("id")) for item in value
            if isinstance(item, dict) and "id" in item and "value" in item and "unit" in item]


def dcg(labels: list[int]) -> float:
    return sum(label / math.log2(idx + 2) for idx, label in enumerate(labels))


def ndcg_at_k(gold_rank: list[str], pred_rank: list[str], k: int = 5) -> float:
    if not gold_rank:
        return 1.0
    gold_top = gold_rank[:k]
    pred_top = pred_rank[:k]
    relevance = {item: len(gold_top) - idx for idx, item in enumerate(gold_top)}
    labels = [relevance.get(item, 0) for item in pred_top]
    ideal = sorted(relevance.values(), reverse=True)
    ideal_dcg = dcg(ideal)
    return 0.0 if ideal_dcg == 0 else dcg(labels) / ideal_dcg


NUMBER = re.compile(r"\d+(?:\.\d+)?%?")


def anomaly_keys(value: Any) -> set[tuple[str, str]]:
    items = value if isinstance(value, list) else [value]
    return {(str(x.get("doc_id")), str(x.get("anomaly_type"))) for x in items if isinstance(x, dict) and x.get("anomaly_type")}


def briefing_consistency(gold_answer: str, pred_answer: str) -> tuple[float, float]:
    gold_facts = set(NUMBER.findall(gold_answer or "")); pred_facts = set(NUMBER.findall(pred_answer or ""))
    consistency = 1.0 if not pred_facts else len(gold_facts & pred_facts) / len(pred_facts)
    hallucination = 0.0 if not pred_facts else len(pred_facts - gold_facts) / len(pred_facts)
    return consistency, hallucination


def dataset2_score(answer_path: Path, pred_path: Path, tolerance: float = 0.01) -> dict[str, float]:
    gold = {row["question_id"]: row for row in read_jsonl(answer_path)}
    pred = {row["question_id"]: row for row in read_jsonl(pred_path)}
    total = len(gold)
    if total == 0:
        return {}
    exact = evidence = 0
    ranking_total = ranking_topk = 0
    ndcg_total = 0.0
    anomaly_tp = anomaly_fp = anomaly_fn = 0
    briefing_total = 0; briefing_consistent = briefing_hallucination = 0.0
    for qid, row in gold.items():
        guess = pred.get(qid, {})
        if values_equal(row.get("answer_value"), guess.get("answer_value"), tolerance):
            exact += 1
        if set(row.get("evidence_rows", [])) == set(guess.get("evidence_rows", [])):
            evidence += 1
        gold_rank = entity_ranking(row.get("answer_value"))
        if gold_rank:
            ranking_total += 1
            pred_rank = entity_ranking(guess.get("answer_value"))
            ranking_topk += int(gold_rank[:5] == pred_rank[:5])
            ndcg_total += ndcg_at_k(gold_rank, pred_rank, 5)
        gold_anom = anomaly_keys(row.get("answer_value")); pred_anom = anomaly_keys(guess.get("answer_value"))
        anomaly_tp += len(gold_anom & pred_anom); anomaly_fp += len(pred_anom - gold_anom); anomaly_fn += len(gold_anom - pred_anom)
        if isinstance(row.get("answer_value"), dict) and "briefing_facts" in row["answer_value"]:
            briefing_total += 1
            consistency, hallucination = briefing_consistency(row.get("final_answer", ""), guess.get("final_answer", ""))
            briefing_consistent += consistency; briefing_hallucination += hallucination
    anomaly_precision = anomaly_tp / (anomaly_tp + anomaly_fp) if anomaly_tp + anomaly_fp else 1.0
    anomaly_recall = anomaly_tp / (anomaly_tp + anomaly_fn) if anomaly_tp + anomaly_fn else 1.0
    anomaly_f1 = 0.0 if anomaly_precision + anomaly_recall == 0 else 2 * anomaly_precision * anomaly_recall / (anomaly_precision + anomaly_recall)
    return {
        "answer_value_accuracy": exact / total,
        "evidence_accuracy": evidence / total,
        "top5_ranking_accuracy": 1.0 if ranking_total == 0 else ranking_topk / ranking_total,
        "ndcg_at_5": 1.0 if ranking_total == 0 else ndcg_total / ranking_total,
        "anomaly_precision": anomaly_precision,
        "anomaly_recall": anomaly_recall,
        "anomaly_f1": anomaly_f1,
        "briefing_factual_consistency": 1.0 if briefing_total == 0 else briefing_consistent / briefing_total,
        "briefing_hallucination_rate": 0.0 if briefing_total == 0 else briefing_hallucination / briefing_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["q", "dataqa"], required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()
    scores = dataset1_score(args.gold, args.pred) if args.dataset == "q" else dataset2_score(args.gold, args.pred, args.tolerance)
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
