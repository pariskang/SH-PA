"""Unit tests for the evaluation runner and provider layer (no network).

Guards the regressions found in the eval-pipeline audit: provider routing,
fact-guard coverage, DataQA output contracts, context scoping, macro
aggregation, preflight, and resume/overwrite file semantics.
"""
import json
from pathlib import Path

import pytest

from gongwen_benchmark.scripts import eval_runner as E
from gongwen_benchmark.scripts import llm_providers as L

ROOT = Path(__file__).resolve().parents[1] / "gongwen_benchmark"


# ── provider routing (C5) ─────────────────────────────────────────────────────

def test_litellm_model_routing():
    # explicit provider prefix → pass through untouched
    assert L.LiteLLMConfig(model="gemini/gemini-2.0-flash").litellm_model == "gemini/gemini-2.0-flash"
    assert L.LiteLLMConfig(model="anthropic/claude-opus-4-8").litellm_model == "anthropic/claude-opus-4-8"
    assert L.LiteLLMConfig(model="together_ai/x/y").litellm_model == "together_ai/x/y"
    # bare id, no relay → pass through so LiteLLM routes by its own model map
    # (forcing openai/ would 404 a bare non-OpenAI id like claude-opus-4-8)
    assert L.LiteLLMConfig(model="gpt-4o", api_base=None).litellm_model == "gpt-4o"
    assert L.LiteLLMConfig(model="claude-opus-4-8", api_base=None).litellm_model == "claude-opus-4-8"
    # bare id against a custom relay endpoint → OpenAI-compatible prefix
    assert L.LiteLLMConfig(model="MiniMax-M1", api_base="https://relay/v1").litellm_model == "openai/MiniMax-M1"


# ── fact guard (M3) ───────────────────────────────────────────────────────────

def test_fact_guard_catches_doctype_swap():
    src = "示范省人民政府关于加强管理的通知"
    assert not L.fact_guard(src, src.replace("通知", "通告"))
    assert L.fact_guard(src, "为加强管理，特发此通知")  # same facts, reworded


def test_fact_guard_catches_agency_code_swap():
    codes = list(L._agency_codes())
    assert len(codes) >= 2, "agency_metadata.csv should yield agency codes"
    a, b = codes[0], codes[1]
    src = f"{a}〔2026〕5号关于某事的通知"
    assert not L.fact_guard(src, src.replace(a, b))


# ── cache key (M5) ────────────────────────────────────────────────────────────

def test_cache_key_includes_temperature():
    msgs = [{"role": "user", "content": "x"}]
    k_lo = L._cache_key(msgs, L.LiteLLMConfig(temperature=0.1))
    k_hi = L._cache_key(msgs, L.LiteLLMConfig(temperature=0.9))
    assert k_lo != k_hi


# ── DataQA / Q output contracts (C1, C2, M2) ──────────────────────────────────

def test_dataqa_message_injects_per_task_schema():
    msgs = E._build_dataqa_messages("Q?", "TABLE", "period_comparison", ["发文量"])
    content = msgs[1]["content"]
    assert "earlier" in content and "later" in content and "change" in content
    assert "period_comparison" in content


def test_dataqa_message_injects_anomaly_codebook():
    msgs = E._build_dataqa_messages("Q?", "TABLE", "anomaly_detection", None)
    content = msgs[1]["content"]
    for code in ("invalid_doc_number", "missing_doc_type_in_title", "secret_on_public"):
        assert code in content


def test_dataqa_ranking_specifies_unit():
    content = E._build_dataqa_messages("Q?", "T", "cross_agency_ranking", None)[1]["content"]
    assert '"unit"' in content and "件" in content


def test_q_system_requests_extra_slots_and_controlled_intents():
    assert "doc_type_name" in E._Q_SYSTEM and "target_doc_type" in E._Q_SYSTEM
    # the controlled intent vocabulary is enumerated for the model
    for intent in ("文种选择", "医疗合规辨析", "安全拒答"):
        assert intent in E._Q_SYSTEM


# ── context scoping (C3) ──────────────────────────────────────────────────────

_RECS = [
    {"doc_id": f"R{i:06d}", "issue_date": d}
    for i, d in enumerate(
        ["2026-03-02", "2026-03-02", "2026-03-03", "2026-03-04"], start=1
    )
]


def test_scope_composite_returns_no_records():
    assert E._scope_records(_RECS, "", "composite_element_explanation") == []


def test_scope_single_date_and_range():
    assert len(E._scope_records(_RECS, "2026-03-02", "direct_lookup")) == 2
    assert len(E._scope_records(_RECS, "2026-03-02~2026-03-03", "period_comparison")) == 3
    assert len(E._scope_records(_RECS, "", "consecutive_compliance_streak")) == 4  # empty → all


def test_scope_row_cap(monkeypatch):
    monkeypatch.setattr(E, "_CONTEXT_ROW_CAP", 2)
    out = E._scope_records(_RECS, "", "consecutive_compliance_streak")
    assert len(out) == 2
    # deterministic order (by issue_date, doc_id)
    assert [r["doc_id"] for r in out] == ["R000001", "R000002"]


# ── macro aggregation (M1) ────────────────────────────────────────────────────

def test_macro_score_equal_weight_excludes_derived_inverts_lower_better():
    scores = {
        "dataset1_q": {"a": 0.8, "b": 0.6},                       # mean 0.7
        "dataset3_writing": {"length_compliance": 1.0, "overall_compliance": 0.0},  # 1.0 (derived excluded)
        "dataset2_dataqa": {"answer_value_accuracy": 0.4, "briefing_hallucination_rate": 0.2},  # (0.4 + 0.8)/2 = 0.6
    }
    assert E._macro_score(scores) == pytest.approx((0.7 + 1.0 + 0.6) / 3, abs=1e-9)


def test_macro_score_counts_int_metrics():
    # int-valued metrics must not be silently dropped
    assert E._macro_score({"d": {"x": 1, "y": 0}}) == pytest.approx(0.5)


# ── preflight (C4) ────────────────────────────────────────────────────────────

def test_preflight_flags_unusable_provider(monkeypatch):
    monkeypatch.setenv("CN_GW_EVAL_PREFLIGHT", "0")  # never reach the live probe
    err = E._preflight(L.LiteLLMConfig(api_key=None))
    assert err  # non-None error string (missing key or missing library)
    err_poe = E._preflight(L.PoeConfig(api_key=None))
    assert err_poe


# ── resume / overwrite file semantics (m5) ────────────────────────────────────

def _patch_llm(monkeypatch):
    monkeypatch.setattr(
        E, "completion_json",
        lambda messages, config: {
            "target_doc_type": "", "expected_query_type": "POLICY_EXPLANATION",
            "requires_clarification": False, "should_refuse": False, "expected_slots": {},
        },
    )
    monkeypatch.setattr(E, "_CALL_DELAY", 0.0)


def _qids(path: Path):
    return [json.loads(l)["question_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_fresh_run_overwrites_and_resume_has_no_duplicates(tmp_path, monkeypatch):
    _patch_llm(monkeypatch)
    out = tmp_path
    # fresh run, 3 questions
    E.evaluate_dataset1(L.LiteLLMConfig(), out, limit=3, resume=False)
    rows = _qids(out / "pred_dataset1_q.jsonl")
    assert len(rows) == 3 and len(set(rows)) == 3

    # resume: all already done → no new rows, no duplicates
    E.evaluate_dataset1(L.LiteLLMConfig(), out, limit=3, resume=True)
    rows2 = _qids(out / "pred_dataset1_q.jsonl")
    assert rows2 == rows  # unchanged, no dupes

    # fresh run again truncates a stale (larger) file rather than appending
    E.evaluate_dataset1(L.LiteLLMConfig(), out, limit=2, resume=False)
    rows3 = _qids(out / "pred_dataset1_q.jsonl")
    assert len(rows3) == 2 and len(set(rows3)) == 2
