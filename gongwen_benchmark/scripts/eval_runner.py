"""Multi-model evaluation runner for CN-GongWen benchmark (E7 Leaderboard).

Runs one or more models on all four datasets, saves per-model predictions, and
compiles a leaderboard with per-task scores and 95% bootstrap confidence intervals.

Quick start
-----------
Single model (LiteLLM):
    python gongwen_benchmark/scripts/eval_runner.py \\
        --provider litellm --model openai/gpt-4o \\
        --dataset all --output-dir results/gpt4o

Azure OpenAI:
    export AZURE_API_KEY=... AZURE_API_BASE=https://xxx.openai.azure.com/
    python gongwen_benchmark/scripts/eval_runner.py \\
        --provider azure --model gpt-4o \\
        --dataset all --output-dir results/gpt4o-azure

Poe:
    export POE_API_KEY=...
    python gongwen_benchmark/scripts/eval_runner.py \\
        --provider poe --model Claude-3-7-Sonnet \\
        --dataset q --output-dir results/claude-poe

Multi-model leaderboard (YAML config):
    python gongwen_benchmark/scripts/eval_runner.py \\
        --models models.yaml --dataset all --output-dir results/

Compile leaderboard from saved predictions:
    python gongwen_benchmark/scripts/eval_runner.py \\
        --leaderboard-only --output-dir results/

models.yaml format
------------------
    - name: GPT-4o
      provider: litellm
      model: openai/gpt-4o
      api_key: sk-...          # optional; falls back to env vars
      api_base: null

    - name: Azure-GPT-4o
      provider: azure
      deployment: gpt-4o
      api_base: https://myresource.openai.azure.com/
      api_key: ...
      api_version: "2024-08-01-preview"

    - name: Claude-via-Poe
      provider: poe
      bot_name: Claude-3-7-Sonnet
      api_key: ...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from llm_providers import (
    AzureConfig,
    LiteLLMConfig,
    PoeConfig,
    ProviderConfig,
    completion_json,
    completion_text,
    config_from_provider,
    litellm_available,
    poe_available,
)

DATASET1 = ROOT / "dataset_1_question_only"
DATASET2 = ROOT / "dataset_2_data_qa"
DATASET3 = ROOT / "dataset_3_writing"
DATASET4 = ROOT / "dataset_4_audit"
EVALUATION = ROOT / "evaluation"

# Delay between API calls to avoid rate limits (seconds).
_CALL_DELAY = float(os.getenv("CN_GW_EVAL_DELAY", "1.0"))
# Bootstrap resampling iterations for CI.
_BOOTSTRAP_N = int(os.getenv("CN_GW_BOOTSTRAP_N", "1000"))
# Max records fed as DataQA context (0 = unlimited). Bound it for small-context
# models; large-scope aggregate questions need a big-context model to stay correct.
_CONTEXT_ROW_CAP = int(os.getenv("CN_GW_CONTEXT_ROW_CAP", "0"))


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _load_records_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ──────────────────────────────────────────────────────────────────────────────
# Context builders for DataQA (Dataset 2)
# ──────────────────────────────────────────────────────────────────────────────

def _scope_records(
    records: list[dict[str, str]], required_scope: str, task_type: str = ""
) -> list[dict[str, str]]:
    """Filter records to the date/range given by required_scope.

    EV_CAP (40) in the generator caps *gold evidence labels*, not LLM input
    context, so by default all scoped records are returned (the model needs them
    all for aggregates). Two refinements keep context tractable:
      * composite_element_explanation is answered from the 发文字号 quoted in the
        question itself, so it needs no records (its scope is empty, which would
        otherwise dump the whole corpus);
      * an optional CN_GW_CONTEXT_ROW_CAP bounds the row count (deterministically
        ordered) for small-context models.
    """
    if task_type == "composite_element_explanation":
        return []
    if not required_scope:
        scoped = records
    elif "~" in required_scope:
        start, end = required_scope.split("~", 1)
        scoped = [r for r in records if start <= r.get("issue_date", "") <= end]
    else:
        date = required_scope.strip()
        scoped = [r for r in records if r.get("issue_date", "") == date]
    if _CONTEXT_ROW_CAP and len(scoped) > _CONTEXT_ROW_CAP:
        scoped = sorted(
            scoped, key=lambda r: (r.get("issue_date", ""), r.get("doc_id", ""))
        )[:_CONTEXT_ROW_CAP]
    return scoped


def _records_to_table(rows: list[dict[str, str]]) -> str:
    """Render a compact CSV-like table string suitable for LLM context."""
    if not rows:
        return "（无记录）"
    keep = [
        "doc_id", "agency_id", "agency_name", "agency_code",
        "doc_type_name", "doc_number", "title", "main_recipient",
        "direction", "security_level", "urgency", "issue_date",
        "has_attachment", "cc_count", "page_count",
        "format_flag", "policy_domain", "medical_area", "medical_topic",
    ]
    headers = [k for k in keep if k in rows[0]]
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(r.get(h, "") for h in headers))
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders — one per dataset
# ──────────────────────────────────────────────────────────────────────────────

_Q_SYSTEM = (
    "你是党政机关公文写作与办理评测助手。请以 JSON 格式回答公文业务问题，"
    "不得包含任何 Markdown 格式或额外说明文字。\n"
    "输出格式（字段均为必填）：\n"
    '{"target_doc_type":"（适用时填 GWxx，如 GW08；不适用或无法确定时填空字符串）",'
    '"expected_query_type":"（从下列之一选择：DOC_TYPE_SELECTION / FORMAT_VALIDATION / '
    "APPLICABILITY_EXPLANATION / DIRECTION_JUDGMENT / POLICY_EXPLANATION / "
    'CLARIFICATION_REQUIRED / SAFE_REFUSAL_REQUIRED）",'
    '"requires_clarification":true 或 false,'
    '"should_refuse":true 或 false,'
    '"expected_slots":{"intent":"<意图，从下列受控词选一>",'
    '"target_doc_type":"<若涉及文种选择，填 GWxx>",'
    '"doc_type_name":"<若涉及文种，填中文文种名，如 通知>"}}\n'
    "intent 受控词（只能选其一，须逐字一致）：文种选择 / 文种辨析 / 文种误用辨析 / "
    "文种与行文方向冲突 / 行文关系规范 / 行文权限边界 / 跨要素合规判断 / 格式边界精度 / "
    "时效与办理时序 / 否定枚举 / 意图澄清 / 管理建议 / 口语化文种识别 / 安全拒答 / 医疗合规辨析。\n"
    "expected_slots 中不适用的键填空字符串；target_doc_type/doc_type_name 仅在涉及文种时填写。"
)

_DATAQA_SYSTEM = (
    "你是公文数据分析专家。以下提供公文档案记录（CSV 格式），请严格根据这些数据"
    "回答数据分析问题，不得推测或引用档案外的信息。\n"
    "输出 JSON（字段均为必填）：\n"
    '{"answer_value":<严格按下方【answer_value 结构】给出>,'
    '"evidence_rows":["R000001",...]（answer_value 所依据的 doc_id 列表）,'
    '"final_answer":"用中文完整回答问题的一到两句话"}\n'
    "注意：answer_value 的键名与结构必须与给定结构完全一致（含中文键名、单位、嵌套层级），"
    "否则无法判分。"
)

# Per-task_type answer_value contract — mirrors exactly the schema the generator
# emits per task_type (verified against answers.jsonl). Without this, a model that
# computes the right numbers writes them under the wrong keys and scores 0.
_DATAQA_SCHEMAS: dict[str, str] = {
    "direct_lookup":
        '{"doc_id":"该公文的Rxxxxxx编号","element":"要素名(如 发文字号/成文日期)","value":"该要素的值"}',
    "cross_agency_ranking":
        '[{"id":"GA机关号","value":发文量数值,"unit":"件"}]（按 value 降序，取前5）',
    "priority_ranking":
        '[{"id":"R公文号","value":优先级分数值,"unit":"优先级分"}]（按 value 降序，取前5）',
    "multi_criteria_ranking":
        '[{"id":"R公文号","value":综合分数值,"unit":"综合分"}]（按 value 降序，取前5）',
    "period_comparison":
        '{"earlier":前一时期数量,"later":后一时期数量,"change":后减前的差值}',
    "sustained_trend":
        '{"series":[各时段数量,...],"trend":"上升|下降|平稳|波动"之一}',
    "composite_element_explanation":
        '{"机关代字":"...","年份":"...","发文顺序号":"..."}（拆解题面给出的发文字号，键名用中文）',
    "anomaly_detection":
        '[{"doc_id":"R公文号","anomaly_type":"下列代码之一"}]（仅列出存在异常的公文）',
    "briefing":
        '{"briefing_facts":{"agency_id":"GAxxx","date":"YYYY-MM-DD","total":总数,'
        '"by_direction":{"下行文":n,"上行文":n,"平行文":n},"violation_count":n,'
        '"urgent_count":n,"secret_count":n}}',
    "cross_doc_extremum":
        '{"doc_id":"R公文号","value":数值,"unit":"题面所述计量单位"}',
    "consecutive_compliance_streak":
        '{"agency_id":"GAxxx","streak_days":连续天数}',
    "counterfactual_format":
        '{"compliant_after_fix":true或false,"remaining_issues":[剩余问题代码列表，无则空]}',
    "quality_filtered_aggregate":
        '{"value":数量,"unit":"件"}',
    "precision_percentage_change":
        '{"value":百分比数值(保留小数),"unit":"%"}',
    "negative_enumeration":
        '{"agencies":["GAxxx",...],"count":数量}',
    "policy_domain_classification":
        '{"policy_domain":"通用政务或医疗卫生","medical_area":"医疗子领域(非医疗则空字符串)",'
        '"medical_topic":"具体政策主题(非医疗则空字符串)"}',
}

# Closed anomaly_type vocabulary (mirrors anomaly_labels.jsonl). Like the Audit
# codebook, the model cannot match the gold snake_case codes unless we list them.
_ANOMALY_CODES: dict[str, str] = {
    "invalid_doc_number": "发文字号格式错误（年份应用六角括号〔〕、序号不加“第”、不编虚位）",
    "missing_doc_type_in_title": "标题省略文种（标题三要素不全）",
    "missing_signatory_upward": "上行文缺少签发人",
    "secret_on_public": "公布性公文面向社会公开，不应标注密级",
    "missing_main_recipient": "上行/下行公文缺少主送机关",
    "multi_head_qingshi": "请示多头主送（违反一文一事、单一主送）",
    "seal_on_jiyao": "纪要不加盖印章",
}

_WRITING_SYSTEM = (
    "你是党政机关公文起草专家，请严格按照《党政机关公文处理工作条例》(2012) 和 "
    "GB/T 9704—2012 起草公文。直接输出完整公文正文，"
    "不要有任何 Markdown 格式、代码块或前言说明。"
)

_AUDIT_SYSTEM = (
    "你是党政机关公文格式审核专家，依据《党政机关公文处理工作条例》与 GB/T 9704—2012 "
    "审核公文格式。只输出 JSON，不要任何 Markdown 或说明文字。\n"
    '输出格式：{"violations":["code1","code2"]}（无问题则返回空列表）'
)


def _build_q_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _Q_SYSTEM},
        {"role": "user", "content": f"请回答以下公文业务问题：\n{question}"},
    ]


def _build_dataqa_messages(
    question: str, context_table: str, task_type: str = "", required_elements=None
) -> list[dict[str, str]]:
    parts: list[str] = []
    schema = _DATAQA_SCHEMAS.get(task_type)
    if schema:
        parts.append(f"【answer_value 结构】（本题 task_type={task_type}）\n{schema}")
    if task_type == "anomaly_detection":
        codebook = "\n".join(f"  - {c}：{d}" for c, d in _ANOMALY_CODES.items())
        parts.append(f"【anomaly_type 取值（闭集，只能用下列代码）】\n{codebook}")
    if required_elements:
        parts.append("【需关注要素】" + "、".join(required_elements))
    if task_type == "composite_element_explanation":
        parts.append("（本题无需档案记录，请依据题面给出的发文字号直接拆解作答。）")
    schema_block = ("\n\n".join(parts) + "\n\n") if parts else ""
    return [
        {"role": "system", "content": _DATAQA_SYSTEM},
        {
            "role": "user",
            "content": f"{schema_block}【档案记录（CSV）】\n{context_table}\n\n【问题】\n{question}",
        },
    ]


def _build_writing_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _WRITING_SYSTEM},
        {"role": "user", "content": prompt},
    ]


def _build_audit_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _AUDIT_SYSTEM},
        {"role": "user", "content": prompt},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset evaluation logic
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_dataset1(
    config: ProviderConfig,
    output_dir: Path,
    verbose: bool = False,
    limit: int | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run Dataset 1 (Q) and return predictions JSONL rows.

    Supports resume: questions already present in the output file are skipped.
    """
    questions = _read_jsonl(DATASET1 / "questions_public.jsonl")
    if limit:
        questions = questions[:limit]
    hidden_rows = _read_jsonl(DATASET1 / "questions_with_hidden_metadata.jsonl")
    # O(1) lookup: question text → question_id (handles public files without IDs).
    _text_to_qid: dict[str, str] = {
        r.get("question", ""): r["question_id"]
        for r in hidden_rows
        if r.get("question_id")
    }

    out_path = output_dir / "pred_dataset1_q.jsonl"
    # Load already-completed predictions for resume support.
    done: dict[str, dict[str, Any]] = {
        r["question_id"]: r
        for r in (_read_jsonl(out_path) if resume else [])
        if r.get("question_id")
    }

    predictions: list[dict[str, Any]] = []
    errors = 0
    with out_path.open("a" if resume else "w", encoding="utf-8") as fh_out:
        for i, row in enumerate(questions):
            qid = (
                row.get("question_id")
                or _text_to_qid.get(row.get("question", ""))
                or f"Q_ONLY_{i+1:06d}"
            )
            if qid in done:
                predictions.append(done[qid])
                continue
            messages = _build_q_messages(row["question"])
            try:
                result = completion_json(messages, config)
                pred = {
                    "question_id": qid,
                    "target_doc_type": result.get("target_doc_type", ""),
                    "expected_query_type": result.get("expected_query_type", ""),
                    "requires_clarification": bool(result.get("requires_clarification", False)),
                    "should_refuse": bool(result.get("should_refuse", False)),
                    "expected_slots": result.get("expected_slots", {}),
                }
            except Exception as exc:
                errors += 1
                if verbose:
                    print(f"  [warn] {qid}: {exc}", file=sys.stderr)
                pred = {
                    "question_id": qid,
                    "target_doc_type": "",
                    "expected_query_type": "",
                    "requires_clarification": False,
                    "should_refuse": False,
                    "expected_slots": {},
                }
            predictions.append(pred)
            fh_out.write(json.dumps(pred, ensure_ascii=False, separators=(",", ":")) + "\n")
            fh_out.flush()
            if verbose and (i + 1) % 50 == 0:
                print(f"  Q: {i+1}/{len(questions)}", file=sys.stderr)
            time.sleep(_CALL_DELAY)
    if errors:
        print(f"  ⚠ {errors} call(s) failed — empty predictions written for those.",
              file=sys.stderr)
    return predictions


def evaluate_dataset2(
    config: ProviderConfig,
    output_dir: Path,
    verbose: bool = False,
    limit: int | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run Dataset 2 (DataQA) with record context and return predictions.

    Supports resume: questions already present in the output file are skipped.
    """
    questions = _read_jsonl(DATASET2 / "questions.jsonl")
    if limit:
        questions = questions[:limit]
    records = _load_records_csv(DATASET2 / "records.csv")

    out_path = output_dir / "pred_dataset2_dataqa.jsonl"
    done: dict[str, dict[str, Any]] = {
        r["question_id"]: r
        for r in (_read_jsonl(out_path) if resume else [])
        if r.get("question_id")
    }

    predictions: list[dict[str, Any]] = []
    errors = 0
    with out_path.open("a" if resume else "w", encoding="utf-8") as fh_out:
        for i, row in enumerate(questions):
            qid = row["question_id"]
            if qid in done:
                predictions.append(done[qid])
                continue
            task_type = row.get("task_type", "")
            scope = row.get("required_scope", "")
            context_rows = _scope_records(records, scope, task_type)
            context_table = _records_to_table(context_rows)
            messages = _build_dataqa_messages(
                row["question"], context_table, task_type, row.get("required_elements")
            )
            try:
                result = completion_json(messages, config)
                pred = {
                    "question_id": qid,
                    "answer_value": result.get("answer_value"),
                    "evidence_rows": result.get("evidence_rows", []),
                    "final_answer": result.get("final_answer", ""),
                }
            except Exception as exc:
                errors += 1
                if verbose:
                    print(f"  [warn] {qid}: {exc}", file=sys.stderr)
                pred = {
                    "question_id": qid,
                    "answer_value": None,
                    "evidence_rows": [],
                    "final_answer": "",
                }
            predictions.append(pred)
            fh_out.write(json.dumps(pred, ensure_ascii=False, separators=(",", ":")) + "\n")
            fh_out.flush()
            if verbose and (i + 1) % 100 == 0:
                print(f"  DataQA: {i+1}/{len(questions)}", file=sys.stderr)
            time.sleep(_CALL_DELAY)
    if errors:
        print(f"  ⚠ {errors} call(s) failed — empty predictions written for those.",
              file=sys.stderr)
    return predictions


def evaluate_dataset3(
    config: ProviderConfig,
    output_dir: Path,
    verbose: bool = False,
    limit: int | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run Dataset 3 (Writing) and return predictions with document text.

    Supports resume: questions already present in the output file are skipped.
    """
    prompts = _read_jsonl(DATASET3 / "writing_prompts_public.jsonl")
    if limit:
        prompts = prompts[:limit]

    out_path = output_dir / "pred_dataset3_writing.jsonl"
    done: dict[str, dict[str, Any]] = {
        r["question_id"]: r
        for r in (_read_jsonl(out_path) if resume else [])
        if r.get("question_id")
    }

    predictions: list[dict[str, Any]] = []
    errors = 0
    with out_path.open("a" if resume else "w", encoding="utf-8") as fh_out:
        for i, row in enumerate(prompts):
            qid = row["question_id"]
            if qid in done:
                predictions.append(done[qid])
                continue
            messages = _build_writing_messages(row["prompt"])
            try:
                text = completion_text(messages, config)
                pred = {"question_id": qid, "answer": text.strip()}
            except Exception as exc:
                errors += 1
                if verbose:
                    print(f"  [warn] {qid}: {exc}", file=sys.stderr)
                pred = {"question_id": qid, "answer": ""}
            predictions.append(pred)
            fh_out.write(json.dumps(pred, ensure_ascii=False, separators=(",", ":")) + "\n")
            fh_out.flush()
            if verbose and (i + 1) % 20 == 0:
                print(f"  Writing: {i+1}/{len(prompts)}", file=sys.stderr)
            time.sleep(_CALL_DELAY)
    if errors:
        print(f"  ⚠ {errors} call(s) failed — empty predictions written for those.",
              file=sys.stderr)
    return predictions


def evaluate_dataset4_audit(
    config: ProviderConfig,
    output_dir: Path,
    verbose: bool = False,
    limit: int | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run Dataset 4 (Audit – find violations) and return predictions.

    Supports resume: questions already present in the output file are skipped.
    """
    tasks = _read_jsonl(DATASET4 / "audit_tasks_public.jsonl")
    if limit:
        tasks = tasks[:limit]

    out_path = output_dir / "pred_dataset4_audit.jsonl"
    done: dict[str, dict[str, Any]] = {
        r["question_id"]: r
        for r in (_read_jsonl(out_path) if resume else [])
        if r.get("question_id")
    }

    predictions: list[dict[str, Any]] = []
    errors = 0
    with out_path.open("a" if resume else "w", encoding="utf-8") as fh_out:
        for i, row in enumerate(tasks):
            qid = row["question_id"]
            if qid in done:
                predictions.append(done[qid])
                continue
            messages = _build_audit_messages(row["prompt"])
            try:
                result = completion_json(messages, config)
                violations = result.get("violations", [])
                if not isinstance(violations, list):
                    violations = []
                pred = {"question_id": qid, "violations": violations}
            except Exception as exc:
                errors += 1
                if verbose:
                    print(f"  [warn] {qid}: {exc}", file=sys.stderr)
                pred = {"question_id": qid, "violations": []}
            predictions.append(pred)
            fh_out.write(json.dumps(pred, ensure_ascii=False, separators=(",", ":")) + "\n")
            fh_out.flush()
            if verbose and (i + 1) % 30 == 0:
                print(f"  Audit: {i+1}/{len(tasks)}", file=sys.stderr)
            time.sleep(_CALL_DELAY)
    if errors:
        print(f"  ⚠ {errors} call(s) failed — empty predictions written for those.",
              file=sys.stderr)
    return predictions


def evaluate_dataset4_rewrite(
    config: ProviderConfig,
    output_dir: Path,
    verbose: bool = False,
    limit: int | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run Dataset 4 (Audit – rewrite/correct flawed document).

    Supports resume: questions already present in the output file are skipped.
    """
    tasks = _read_jsonl(DATASET4 / "audit_tasks_public.jsonl")
    if limit:
        tasks = tasks[:limit]

    out_path = output_dir / "pred_dataset4_rewrite.jsonl"
    done: dict[str, dict[str, Any]] = {
        r["question_id"]: r
        for r in (_read_jsonl(out_path) if resume else [])
        if r.get("question_id")
    }

    predictions: list[dict[str, Any]] = []
    errors = 0
    with out_path.open("a" if resume else "w", encoding="utf-8") as fh_out:
        for i, row in enumerate(tasks):
            qid = row["question_id"]
            rewrite_prompt = row.get("rewrite_prompt", "")
            if not rewrite_prompt:
                continue
            if qid in done:
                predictions.append(done[qid])
                continue
            messages = _build_writing_messages(rewrite_prompt)
            try:
                text = completion_text(messages, config)
                pred = {"question_id": qid, "rewrite": text.strip()}
            except Exception as exc:
                errors += 1
                if verbose:
                    print(f"  [warn] {qid}: {exc}", file=sys.stderr)
                pred = {"question_id": qid, "rewrite": ""}
            predictions.append(pred)
            fh_out.write(json.dumps(pred, ensure_ascii=False, separators=(",", ":")) + "\n")
            fh_out.flush()
            if verbose and (i + 1) % 30 == 0:
                print(f"  Rewrite: {i+1}/{len(tasks)}", file=sys.stderr)
            time.sleep(_CALL_DELAY)
    if errors:
        print(f"  ⚠ {errors} call(s) failed — empty predictions written for those.",
              file=sys.stderr)
    return predictions


# ──────────────────────────────────────────────────────────────────────────────
# Scoring and leaderboard
# ──────────────────────────────────────────────────────────────────────────────

def _score_model(model_dir: Path) -> dict[str, Any]:
    """Score all available predictions in model_dir using scorer.py and return results."""
    sys.path.insert(0, str(EVALUATION))
    try:
        from scorer import (  # type: ignore[import]
            dataset1_score,
            dataset2_score,
            dataset3_writing_score,
            dataset4_audit_score,
            dataset4_rewrite_score,
        )
    finally:
        sys.path.pop(0)

    results: dict[str, Any] = {}

    p1 = model_dir / "pred_dataset1_q.jsonl"
    if p1.exists():
        results["dataset1_q"] = dataset1_score(
            DATASET1 / "questions_with_hidden_metadata.jsonl", p1
        )

    p2 = model_dir / "pred_dataset2_dataqa.jsonl"
    if p2.exists():
        results["dataset2_dataqa"] = dataset2_score(DATASET2 / "answers.jsonl", p2)

    p3 = model_dir / "pred_dataset3_writing.jsonl"
    if p3.exists():
        results["dataset3_writing"] = dataset3_writing_score(
            DATASET3 / "writing_prompts_with_rubric.jsonl", p3
        )

    p4a = model_dir / "pred_dataset4_audit.jsonl"
    if p4a.exists():
        results["dataset4_audit"] = dataset4_audit_score(
            DATASET4 / "audit_tasks_with_gold.jsonl", p4a
        )

    p4r = model_dir / "pred_dataset4_rewrite.jsonl"
    if p4r.exists():
        results["dataset4_rewrite"] = dataset4_rewrite_score(
            DATASET4 / "audit_tasks_with_gold.jsonl", p4r
        )

    return results


def _bootstrap_ci(values: list[float], n: int = _BOOTSTRAP_N) -> tuple[float, float]:
    """Return 95% bootstrap CI (lo, hi) for the mean of values."""
    if not values:
        return (0.0, 0.0)
    means = []
    rng = random.Random(42)
    for _ in range(n):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    return (lo, hi)


# Metrics where lower is better — inverted (1 - x) before averaging.
_LOWER_IS_BETTER = {"briefing_hallucination_rate"}
# Derived/summary metrics excluded from macro (they would double-count components).
_DERIVED_METRICS = {"overall_compliance"}


def _macro_score(scores: dict[str, Any]) -> float:
    """Equal-weight mean across scoring paths.

    Each scoring path (q / dataqa / writing / audit / rewrite) contributes the
    mean of its own leaf metrics, then those per-path means are averaged with
    equal weight — so a dataset's influence does not scale with how many metrics
    it happens to report. Lower-is-better metrics are inverted; derived/summary
    metrics that would double-count their components are excluded.
    """
    path_means = []
    for dataset_scores in scores.values():
        if not isinstance(dataset_scores, dict):
            continue
        leaf = []
        for key, v in dataset_scores.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if key in _DERIVED_METRICS:
                continue
            leaf.append((1.0 - float(v)) if key in _LOWER_IS_BETTER else float(v))
        if leaf:
            path_means.append(sum(leaf) / len(leaf))
    return sum(path_means) / len(path_means) if path_means else 0.0


def compile_leaderboard(output_dir: Path) -> dict[str, Any]:
    """Scan output_dir for per-model subdirs, score each, produce leaderboard.json."""
    entries = []
    skipped = []
    for model_dir in sorted(output_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        meta_path = model_dir / "model_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("preflight_error"):
            # A model whose provider was unusable wrote no real predictions; do
            # not score it (empty predictions would otherwise rank as ~0.57).
            print(f"  Skipping {meta.get('name', model_dir.name)} (preflight failed) …",
                  file=sys.stderr)
            skipped.append({"model": meta, "reason": meta["preflight_error"]})
            continue
        print(f"  Scoring {meta.get('name', model_dir.name)} …", file=sys.stderr)
        scores = _score_model(model_dir)
        macro = _macro_score(scores)
        entries.append({"model": meta, "scores": scores, "macro_avg": macro})

    entries.sort(key=lambda e: e["macro_avg"], reverse=True)
    for rank, entry in enumerate(entries, 1):
        entry["rank"] = rank

    # Dispersion of macro_avg BETWEEN models (between-model spread), only
    # meaningful with ≥2 models — NOT a per-item confidence interval.
    if len(entries) >= 2:
        lo, hi = _bootstrap_ci([e["macro_avg"] for e in entries])
        spread: dict[str, float] | None = {"macro_avg_lo": lo, "macro_avg_hi": hi}
    else:
        spread = None

    board = {
        "leaderboard": entries,
        "skipped": skipped,
        "macro_avg_spread_across_models_95": spread,
        "note": (
            "macro_avg = equal-weight mean across scoring paths "
            "(q/dataqa/writing/audit/rewrite); each path = mean of its leaf "
            "metrics; lower-is-better metrics inverted; derived metrics "
            "(overall_compliance) excluded. macro_avg_spread_across_models_95 is "
            "the bootstrap dispersion of macro_avg BETWEEN models (n=1000), not "
            "per-item confidence; null for <2 models. Models that failed preflight "
            "are listed under 'skipped', not ranked."
        ),
    }
    out_path = output_dir / "leaderboard.json"
    out_path.write_text(
        json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Leaderboard saved → {out_path}", file=sys.stderr)
    return board


# ──────────────────────────────────────────────────────────────────────────────
# Config loading (YAML or inline)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    provider: str
    model: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    # Azure-specific
    deployment: str | None = None
    api_version: str | None = None
    # Poe-specific
    bot_name: str | None = None

    def to_provider_config(self) -> ProviderConfig:
        p = self.provider.lower()
        if p == "azure":
            return AzureConfig(
                deployment=self.deployment or self.model or "gpt-4o",
                api_base=self.api_base or os.getenv("AZURE_API_BASE"),
                api_key=self.api_key or os.getenv("AZURE_API_KEY"),
                api_version=self.api_version or os.getenv("AZURE_API_VERSION", "2024-08-01-preview"),
            )
        if p == "poe":
            return PoeConfig(
                bot_name=self.bot_name or self.model or os.getenv("POE_BOT_NAME", "GPT-4o-Mini"),
                api_key=self.api_key or os.getenv("POE_API_KEY"),
            )
        # litellm (default)
        cfg_kwargs: dict[str, Any] = {}
        if self.model:
            cfg_kwargs["model"] = self.model
        if self.api_key:
            cfg_kwargs["api_key"] = self.api_key
        if self.api_base:
            cfg_kwargs["api_base"] = self.api_base
        return LiteLLMConfig(**cfg_kwargs) if cfg_kwargs else LiteLLMConfig()


def _load_models_yaml(path: Path) -> list[ModelSpec]:
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        raise SystemExit(
            "PyYAML is required for --models. Install: pip install 'cn-gongwen-benchmark[dev]'"
        )
    with path.open(encoding="utf-8") as fh:
        entries = yaml.safe_load(fh)
    specs = []
    for e in entries:
        specs.append(
            ModelSpec(
                name=e.get("name", e.get("model", "unknown")),
                provider=e.get("provider", "litellm"),
                model=e.get("model"),
                api_key=e.get("api_key"),
                api_base=e.get("api_base"),
                deployment=e.get("deployment"),
                api_version=e.get("api_version"),
                bot_name=e.get("bot_name"),
            )
        )
    return specs


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

_DATASET_CHOICES = ("q", "dataqa", "writing", "audit", "rewrite", "all")


def _preflight(config: ProviderConfig) -> str | None:
    """Return an error string if the provider is unusable, else None.

    Catches the #1 cause of a silently all-zero run — missing library, missing
    API key, bad endpoint/key — BEFORE any dataset is evaluated, so a broken
    environment fails loudly instead of writing empty predictions that score ~0.
    Set CN_GW_EVAL_PREFLIGHT=0 to skip the (1-call) live probe.
    """
    if isinstance(config, PoeConfig):
        if not poe_available():
            return "fastapi-poe not installed. Run: pip install 'cn-gongwen-benchmark[poe]'"
        if not config.api_key:
            return "POE_API_KEY is not set."
    else:
        if not litellm_available():
            return "litellm not installed. Run: pip install 'cn-gongwen-benchmark[llm]'"
        if not config.api_key:
            return "No API key (set LLM_API_KEY / OPENAI_API_KEY / AZURE_API_KEY / …)."
        if isinstance(config, AzureConfig) and not config.api_base:
            return "Azure endpoint missing. Set AZURE_API_BASE."
    if os.getenv("CN_GW_EVAL_PREFLIGHT", "1") == "0":
        return None
    try:
        completion_text([{"role": "user", "content": "ping"}], config)
    except Exception as exc:  # noqa: BLE001
        return f"probe call failed: {exc}"
    return None


def run_model(
    spec: ModelSpec,
    datasets: list[str],
    base_output_dir: Path,
    verbose: bool = False,
    limit: int | None = None,
    resume: bool = False,
) -> bool:
    """Evaluate one model on the specified datasets. Returns True if it ran."""
    safe_name = spec.name.replace("/", "-").replace(":", "-").replace(" ", "_")
    model_dir = base_output_dir / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)

    config = spec.to_provider_config()
    meta = {
        "name": spec.name,
        "provider": spec.provider,
        "model": spec.model,
        "deployment": spec.deployment,
        "bot_name": spec.bot_name,
    }
    err = _preflight(config)
    if err:
        meta["preflight_error"] = err
        (model_dir / "model_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{spec.name}] ✗ PREFLIGHT FAILED: {err}", file=sys.stderr)
        print(f"[{spec.name}] skipped — no predictions written.", file=sys.stderr)
        return False
    (model_dir / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    do_all = "all" in datasets
    print(f"[{spec.name}] Starting evaluation …", file=sys.stderr)

    if do_all or "q" in datasets:
        print(f"  Dataset 1 (Q) …", file=sys.stderr)
        evaluate_dataset1(config, model_dir, verbose=verbose, limit=limit, resume=resume)

    if do_all or "dataqa" in datasets:
        print(f"  Dataset 2 (DataQA) …", file=sys.stderr)
        evaluate_dataset2(config, model_dir, verbose=verbose, limit=limit, resume=resume)

    if do_all or "writing" in datasets:
        print(f"  Dataset 3 (Writing) …", file=sys.stderr)
        evaluate_dataset3(config, model_dir, verbose=verbose, limit=limit, resume=resume)

    if do_all or "audit" in datasets:
        print(f"  Dataset 4 (Audit) …", file=sys.stderr)
        evaluate_dataset4_audit(config, model_dir, verbose=verbose, limit=limit, resume=resume)

    if do_all or "rewrite" in datasets:
        print(f"  Dataset 4 (Rewrite) …", file=sys.stderr)
        evaluate_dataset4_rewrite(config, model_dir, verbose=verbose, limit=limit, resume=resume)

    print(f"[{spec.name}] Done → {model_dir}", file=sys.stderr)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CN-GongWen benchmark multi-model evaluation runner (E7 Leaderboard).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Single-model mode
    parser.add_argument(
        "--provider",
        choices=["litellm", "azure", "poe"],
        default="litellm",
        help="LLM provider backend (default: litellm)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model string for litellm (e.g. openai/gpt-4o, anthropic/claude-3-5-sonnet-20241022, "
            "qwen/qwen-max, deepseek/deepseek-chat), deployment name for azure, or bot name for poe."
        ),
    )
    parser.add_argument("--api-key", default=None, help="API key (overrides env var)")
    parser.add_argument("--api-base", default=None, help="Custom API base URL")
    parser.add_argument("--api-version", default=None, help="Azure API version")
    parser.add_argument("--name", default=None, help="Human-readable model name for the leaderboard")

    # Multi-model mode
    parser.add_argument(
        "--models",
        type=Path,
        default=None,
        help="YAML file listing multiple models to evaluate in sequence",
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        choices=list(_DATASET_CHOICES),
        default="all",
        help="Which dataset(s) to evaluate (default: all)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for predictions and leaderboard (default: results/)",
    )

    parser.add_argument("--leaderboard-only", action="store_true",
                        help="Skip evaluation; only compile leaderboard from existing predictions")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N items per dataset (smoke test / cost control)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume into an existing output dir: keep already-answered "
                             "questions and only fill gaps (default: overwrite fresh)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-question warnings to stderr")

    args = parser.parse_args()
    datasets = [args.dataset]

    if args.leaderboard_only:
        board = compile_leaderboard(args.output_dir)
        print(json.dumps(board, ensure_ascii=False, indent=2))
        return

    if args.models:
        specs = _load_models_yaml(args.models)
    else:
        # Single-model mode — build a ModelSpec from CLI args.
        spec = ModelSpec(
            name=args.name or (args.model or args.provider),
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
            api_version=args.api_version,
            deployment=args.model if args.provider == "azure" else None,
            bot_name=args.model if args.provider == "poe" else None,
        )
        specs = [spec]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ran_any = False
    for spec in specs:
        ran = run_model(
            spec, datasets, args.output_dir,
            verbose=args.verbose, limit=args.limit, resume=args.resume,
        )
        ran_any = ran_any or ran

    if not ran_any:
        print("All models failed preflight; no evaluation performed.", file=sys.stderr)
        raise SystemExit(1)

    if len(specs) > 1:
        print("\nCompiling leaderboard …", file=sys.stderr)
        board = compile_leaderboard(args.output_dir)
        print(json.dumps(board, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
