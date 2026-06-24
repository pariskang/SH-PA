import json
from pathlib import Path
import pytest

from gongwen_benchmark.scripts.data_sources import ingest_csv
from gongwen_benchmark.scripts.litellm_minimax import fact_guard
from gongwen_benchmark.scripts.validate_artifacts import validate
from gongwen_benchmark.evaluation.scorer import (
    dataset1_score, dataset2_score, dataset3_writing_score, dataset4_audit_score, dataset4_rewrite_score,
)
from gongwen_benchmark.scripts.generate_writing_prompts import build_writing_specs, LENGTH_BUCKETS
from gongwen_benchmark.scripts.tokens import estimate_tokens

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


def test_llm_rewrite_falls_back_without_key(monkeypatch):
    """无密钥/调用失败时，--use-litellm 的改写/润色须优雅回退到确定性模板（冻结流程不中断）。"""
    from gongwen_benchmark.scripts.litellm_minimax import rewrite_question, polish_briefing
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert rewrite_question("原始问题文本", {"question_type": "single_doc_type"}) == "原始问题文本"
    assert polish_briefing("原始播报文本", []) == "原始播报文本"


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


# --- CN-GongWen-Writing（dataset_3）：按目标产出 token 分桶的公文写作测试 prompt ---
def test_medical_compliance_question_type():
    """医疗合规辨析题须存在、恒为医疗方向且为高难。"""
    hidden = [json.loads(l) for l in (ROOT / "dataset_1_question_only/questions_with_hidden_metadata.jsonl").open(encoding="utf-8")]
    mc = [h for h in hidden if h["question_type"] == "medical_compliance"]
    assert len(mc) >= 20
    assert all(h["policy_domain"] == "医疗卫生" for h in mc)
    assert all(h["difficulty"] == "hard" for h in mc)


def test_writing_dataset_buckets_and_coverage():
    report = validate(ROOT)
    assert report["writing"] >= 90
    assert set(report["writing_buckets"]) == {"short", "medium", "long"}
    assert len(set(report["writing_buckets"].values())) == 1  # standard 档三桶均衡
    assert report["writing_doc_type_coverage"] == 15           # 覆盖全部 15 法定文种
    assert report["writing_reference_in_range_share"] == 1.0   # 参考公文全部命中目标 token 区间
    assert report["writing_prompt_grounded_share"] == 1.0      # 测试 prompt 事实接地
    assert 0.45 <= report["writing_medical_share"] <= 0.55     # 约一半医疗政策方向


def test_writing_scorer_is_perfect_on_reference():
    path = ROOT / "dataset_3_writing/writing_prompts_with_rubric.jsonl"
    scores = dataset3_writing_score(path, path)
    assert scores["overall_compliance"] == 1.0
    for dim in ("length", "title", "structure", "closing", "signatory", "recipient", "directional",
                "executability", "punctuation", "language_safety", "trap_avoidance"):
        assert scores[f"{dim}_compliance"] == 1.0


def test_writing_scorer_discriminates_bad_submissions(tmp_path):
    """打分器须能区分不合格公文：长度超界、缺署名/日期、夸大用语、标点错误等应失分。"""
    from dataclasses import asdict
    from gongwen_benchmark.scripts.generate_writing_prompts import (
        build_writing_specs, build_rubric, build_framework, deterministic_reference,
    )
    spec = next(s for s in build_writing_specs(90) if s.doc_type == "通知" and s.direction == "downward")
    gold_row = {"question_id": spec.spec_id, "rubric": build_rubric(spec), "spec": asdict(spec),
                "reference_answer": deterministic_reference(spec, build_framework(spec))}
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps(gold_row, ensure_ascii=False) + "\n", encoding="utf-8")
    # 故意写差：标题无三要素、无层次序号、无署名/日期、含夸大词、第二层序号后误加顿号、远短于目标
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"question_id": spec.spec_id,
        "answer": "关于工作的说明\n（一）、全方位赋能，绝对领先，完美闭环。"}, ensure_ascii=False) + "\n", encoding="utf-8")
    s = dataset3_writing_score(gold, bad)
    assert s["overall_compliance"] < 0.5
    assert s["language_safety_compliance"] == 0.0   # 含夸大/网络化词
    assert s["punctuation_compliance"] == 0.0        # “（一）、”误用且缺六角括号字号
    assert s["closing_compliance"] == 0.0            # 无署名/成文日期
    assert s["length_compliance"] == 0.0             # 远短于目标 token 区间


def test_writing_public_split_has_no_gold():
    row = json.loads((ROOT / "dataset_3_writing/writing_prompts_public.jsonl").open(encoding="utf-8").readline())
    assert set(row) <= {"question_id", "prompt"}
    assert "reference_answer" not in row and "rubric" not in row


def test_writing_specs_obey_xingwen_rules():
    """生成的写作规格须遵循行文方向与请示/函/上行文等硬规则。"""
    from gongwen_benchmark.scripts.benchmark_schema import doc_type_by_name
    for spec in build_writing_specs(90):
        legal = doc_type_by_name(spec.doc_type)
        assert spec.direction in {"upward", "downward", "parallel"}
        if legal.direction != "flexible":
            assert spec.direction == legal.direction          # 非灵活文种须与法定方向一致
        if spec.doc_type == "请示":                            # 请示：单一主送、须签发人
            assert spec.single_recipient and spec.needs_signatory
            assert "各" not in spec.recipient and "、" not in spec.recipient
        if spec.direction == "upward":                         # 上行文不抄送下级
            assert not spec.has_cc
        if spec.doc_type == "函":                              # 函为平行文
            assert spec.direction == "parallel"


# --- CN-GongWen-Audit（dataset_4）：公文审核/纠错 ---
def test_audit_dataset_integrity():
    report = validate(ROOT)
    assert report["audit"] >= 90
    assert report["audit_clean"] > 0 and report["audit_flawed"] > 0
    assert report["audit_violation_coverage"] == 16   # 11 通用 + 5 医疗专属违规均有覆盖


def test_audit_covers_medical_violations():
    import collections
    hidden = [json.loads(l) for l in (ROOT / "dataset_4_audit/audit_tasks_with_gold.jsonl").open(encoding="utf-8")]
    cov = collections.Counter(v for h in hidden for v in h["violations"])
    for med in ("overclaim_cure", "patient_privacy_leak", "research_as_clinical",
                "ai_replaces_physician", "medical_insurance_fraud"):
        assert cov[med] > 0, f"medical violation {med} not covered"


def test_audit_scorer_perfect_on_gold():
    path = ROOT / "dataset_4_audit/audit_tasks_with_gold.jsonl"
    s = dataset4_audit_score(path, path)
    assert s["violation_f1"] == 1.0 and s["exact_match_rate"] == 1.0 and s["clean_doc_accuracy"] == 1.0


def test_audit_public_split_has_no_gold():
    row = json.loads((ROOT / "dataset_4_audit/audit_tasks_public.jsonl").open(encoding="utf-8").readline())
    assert set(row) <= {"question_id", "prompt", "rewrite_prompt"}
    assert "violations" not in row and "corrected_document" not in row


def test_audit_scorer_discriminates(tmp_path):
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({"question_id": "AU_1", "violations": ["hype_language", "year_square_bracket"]}, ensure_ascii=False) + "\n"
        + json.dumps({"question_id": "AU_2", "violations": []}, ensure_ascii=False) + "\n", encoding="utf-8")
    pred = tmp_path / "pred.jsonl"  # 漏报一项、误报一项，且对合规件过度报警
    pred.write_text(
        json.dumps({"question_id": "AU_1", "violations": ["hype_language", "seq_add_di"]}, ensure_ascii=False) + "\n"
        + json.dumps({"question_id": "AU_2", "violations": ["date_chinese"]}, ensure_ascii=False) + "\n", encoding="utf-8")
    s = dataset4_audit_score(gold, pred)
    assert s["violation_f1"] < 1.0
    assert s["exact_match_rate"] == 0.0
    assert s["clean_doc_accuracy"] == 0.0   # 误报合规件


def test_audit_rewrite_scorer_perfect_on_gold():
    path = ROOT / "dataset_4_audit/audit_tasks_with_gold.jsonl"
    s = dataset4_rewrite_score(path, path)   # 金标准=合规底稿 corrected_document
    assert s["violations_removed_rate"] == 1.0 and s["facts_preserved_rate"] == 1.0
    assert s["format_valid_rate"] == 1.0 and s["overall_rewrite_compliance"] == 1.0


def test_audit_rewrite_scorer_discriminates(tmp_path):
    """把含缺陷原文原样当“改写”提交，违规未清除 → 改写分应为 0。"""
    hidden = [json.loads(l) for l in (ROOT / "dataset_4_audit/audit_tasks_with_gold.jsonl").open(encoding="utf-8")]
    flawed_item = next(h for h in hidden if not h["is_clean"])
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps(flawed_item, ensure_ascii=False) + "\n", encoding="utf-8")
    pred = tmp_path / "pred.jsonl"
    pred.write_text(json.dumps({"question_id": flawed_item["question_id"],
                                "rewrite": flawed_item["flawed_document"]}, ensure_ascii=False) + "\n", encoding="utf-8")
    s = dataset4_rewrite_score(gold, pred)
    assert s["violations_removed_rate"] == 0.0 and s["overall_rewrite_compliance"] == 0.0


def test_token_buckets_ordered_and_estimator_counts_cjk():
    assert estimate_tokens("") == 0
    assert estimate_tokens("通知") == 2                         # CJK 每字计 1
    short_hi = LENGTH_BUCKETS["short"]["target_tokens"][1]
    medium_lo, medium_hi = LENGTH_BUCKETS["medium"]["target_tokens"]
    long_lo = LENGTH_BUCKETS["long"]["target_tokens"][0]
    assert short_hi <= medium_lo and medium_hi <= long_lo      # 短≤中≤长，区间不交叠
