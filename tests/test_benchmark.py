import json
from pathlib import Path
import pytest

from gongwen_benchmark.scripts.data_sources import ingest_csv
from gongwen_benchmark.scripts.litellm_minimax import fact_guard
from gongwen_benchmark.scripts.validate_artifacts import validate
from gongwen_benchmark.evaluation.scorer import dataset1_score, dataset2_score

ROOT = Path(__file__).resolve().parents[1] / "gongwen_benchmark"


def test_committed_artifacts_are_cross_file_valid():
    report = validate(ROOT)
    assert report["q"] >= 600
    assert report["dataqa"] >= 1000
    assert report["records"] >= 37 * 8 * 8
    # 进阶难题任务类型必须齐备（8 基线 + 7 进阶）。
    for task in (
        "cross_doc_extremum", "consecutive_compliance_streak", "counterfactual_format",
        "quality_filtered_aggregate", "negative_enumeration", "multi_criteria_ranking",
        "precision_percentage_change", "policy_domain_classification",
    ):
        assert task in report["task_types"]
    # Q 须偏向高难（校验器阈值 40%）。
    assert report["q_hard_share"] >= 0.40
    # 幻觉陷阱须覆盖 ≥12 种不同 trap_type。
    assert report["q_trap_diversity"] >= 12
    # 播报须含三种高级子类型。
    for subtype in ("risk_focused_targeted", "conflicting_signals_briefing", "exclusion_briefing"):
        assert subtype in report["briefing_subtypes"]
    # 播报事实接地：零越界引用。
    assert report["briefing_ungrounded_count"] == 0
    # padding 占比须在预算内。
    assert report["q_padding_share"] <= 0.40
    # 约一半内容须为医疗卫生政策方向（语料与 Q 均在 45%~55%）。
    assert 0.45 <= report["corpus_medical_share"] <= 0.55
    assert 0.45 <= report["q_medical_share"] <= 0.55
    # 医疗政策须覆盖足够多的子领域（共 16 个，要求 ≥14）。
    assert report["medical_area_coverage"] >= 14
    # 医疗子领域须细化为足够多的“十分具体”政策主题（共约 105 个，standard 档应覆盖 ≥90）。
    assert report["medical_topic_coverage"] >= 90
    # 分类任务须同时含医疗与通用两类。
    assert 0 < report["classification_medical"] < report["classification_total"]


def test_medical_subjects_cover_all_areas():
    """语料应覆盖全部 16 个医疗政策子领域。"""
    import csv
    from gongwen_benchmark.scripts.benchmark_schema import MEDICAL_AREAS
    rows = list(csv.DictReader((ROOT / "dataset_2_data_qa/records.csv").open(encoding="utf-8")))
    seen = {r["medical_area"] for r in rows if r["medical_area"]}
    assert seen == {a.name for a in MEDICAL_AREAS}


def test_medical_topic_taxonomy_is_granular():
    """医疗子领域须细化为 ≥100 个互不重复的“十分具体”政策主题，每个子领域 ≥4 个。"""
    from gongwen_benchmark.scripts.benchmark_schema import MEDICAL_AREAS, all_medical_topics
    topics = all_medical_topics()
    assert len(topics) >= 100
    assert len(set(topics)) == len(topics)
    assert all(len(area.topics) >= 4 for area in MEDICAL_AREAS)


def test_committed_corpus_uses_only_taxonomy_topics():
    """语料中出现的具体主题必须全部来自 schema 定义（无越界/拼写漂移）。"""
    import csv
    from gongwen_benchmark.scripts.benchmark_schema import all_medical_topics
    rows = list(csv.DictReader((ROOT / "dataset_2_data_qa/records.csv").open(encoding="utf-8")))
    used = {r["medical_topic"] for r in rows if r["medical_topic"]}
    assert used <= set(all_medical_topics())


def test_question_public_split_has_no_hidden_labels():
    row = json.loads((ROOT / "dataset_1_question_only/questions_public.jsonl").open(encoding="utf-8").readline())
    assert set(row) == {"question"}


def test_fact_guard_rejects_factual_drift():
    source = "示市财〔2026〕12号公文为加急件"
    assert fact_guard(source, "加急件，示市财〔2026〕12号公文已印发")
    assert not fact_guard(source, "示市财〔2026〕13号公文为加急件")


def test_real_ingest_rejects_privacy_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "agency_name,身份证号,doc_type,title,issue_date,direction\n"
        "某机关,310100190001011234,通知,关于X的通知,2026-03-02,downward\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="privacy"):
        ingest_csv(path)


def test_real_ingest_anonymizes_aggregate_rows(tmp_path):
    path = tmp_path / "ok.csv"
    path.write_text(
        "agency_name,doc_type,title,issue_date,direction\n"
        "某某市财政局,通知,关于做好预算管理工作的通知,2026-03-02,downward\n",
        encoding="utf-8",
    )
    rows = ingest_csv(path)
    assert rows[0]["agency_id"].startswith("GA") and rows[0]["source_type"] == "real"


def test_scorer_is_perfect_on_gold():
    hidden = ROOT / "dataset_1_question_only/questions_with_hidden_metadata.jsonl"
    answers = ROOT / "dataset_2_data_qa/answers.jsonl"
    s1 = dataset1_score(hidden, hidden)
    assert s1["doc_type_routing_accuracy"] == 1.0
    assert s1["query_type_accuracy"] == 1.0
    assert s1["safe_refusal_accuracy"] == 1.0
    assert s1["hallucination_resistance_rate"] == 1.0
    s2 = dataset2_score(answers, answers)
    assert s2["answer_value_accuracy"] == 1.0
    assert s2["evidence_accuracy"] == 1.0
    assert s2["anomaly_f1"] == 1.0


def test_repository_does_not_commit_binary_dataset_artifacts():
    assert not (ROOT / "element_dictionary.xlsx").exists()
    assert not (ROOT / "dataset_2_data_qa/records.parquet").exists()
