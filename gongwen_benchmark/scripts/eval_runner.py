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
)

DATASET1 = ROOT / "dataset_1_question_only"
DATASET2 = ROOT / "dataset_2_data_qa"
DATASET3 = ROOT / "dataset_3_writing"
DATASET4 = ROOT / "dataset_4_audit"
EVALUATION = ROOT / "evaluation"

# Maximum records to include in a DataQA context window (mirrors EV_CAP in generator).
_CONTEXT_ROW_CAP = 40
# Delay between API calls to avoid rate limits (seconds).
_CALL_DELAY = float(os.getenv("CN_GW_EVAL_DELAY", "1.0"))
# Bootstrap resampling iterations for CI.
_BOOTSTRAP_N = int(os.getenv("CN_GW_BOOTSTRAP_N", "1000"))


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
    records: list[dict[str, str]], required_scope: str
) -> list[dict[str, str]]:
    """Filter records to the date/range given by required_scope."""
    if not required_scope:
        return records[:_CONTEXT_ROW_CAP]
    if "~" in required_scope:
        start, end = required_scope.split("~", 1)
        filtered = [r for r in records if start <= r.get("issue_date", "") <= end]
    else:
        date = required_scope.strip()
        filtered = [r for r in records if r.get("issue_date", "") == date]
    return filtered[:_CONTEXT_ROW_CAP]


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
    '"expected_slots":{"intent":"（一句话描述问题意图）"}}'
)

_DATAQA_SYSTEM = (
    "你是公文数据分析专家。以下提供公文档案记录（CSV 格式），请严格根据这些数据"
    "回答数据分析问题，不得推测或引用档案外的信息。\n"
    "输出 JSON 格式（字段均为必填）：\n"
    '{"answer_value":（具体答案：数值/排名列表/字典/布尔值，类型视问题而定）,'
    '"evidence_rows":["R000001",...],'
    '"final_answer":"用中文完整回答问题的一到两句话"}'
)

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
    question: str, context_table: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _DATAQA_SYSTEM},
        {
            "role": "user",
            "content": f"【档案记录（CSV）】\n{context_table}\n\n【问题】\n{question}",
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
) -> list[dict[str, Any]]:
    """Run Dataset 1 (Q) and return predictions JSONL rows."""
    questions = _read_jsonl(DATASET1 / "questions_public.jsonl")
    hidden = {
        r["question_id"]: r
        for r in _read_jsonl(DATASET1 / "questions_with_hidden_metadata.jsonl")
    }
    predictions: list[dict[str, Any]] = []
    for i, row in enumerate(questions):
        qid = hidden.get(row.get("question_id", f"Q_ONLY_{i+1:06d}"), {}).get(
            "question_id", f"Q_ONLY_{i+1:06d}"
        )
        # Tolerate public file that lacks question_id.
        for meta_row in hidden.values():
            if meta_row.get("question") == row.get("question"):
                qid = meta_row["question_id"]
                break
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
        if verbose and (i + 1) % 50 == 0:
            print(f"  Q: {i+1}/{len(questions)}", file=sys.stderr)
        time.sleep(_CALL_DELAY)
    _write_jsonl(output_dir / "pred_dataset1_q.jsonl", predictions)
    return predictions


def evaluate_dataset2(
    config: ProviderConfig,
    output_dir: Path,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Run Dataset 2 (DataQA) with record context and return predictions."""
    questions = _read_jsonl(DATASET2 / "questions.jsonl")
    records = _load_records_csv(DATASET2 / "records.csv")
    predictions: list[dict[str, Any]] = []
    for i, row in enumerate(questions):
        qid = row["question_id"]
        scope = row.get("required_scope", "")
        context_rows = _scope_records(records, scope)
        context_table = _records_to_table(context_rows)
        messages = _build_dataqa_messages(row["question"], context_table)
        try:
            result = completion_json(messages, config)
            pred = {
                "question_id": qid,
                "answer_value": result.get("answer_value"),
                "evidence_rows": result.get("evidence_rows", []),
                "final_answer": result.get("final_answer", ""),
            }
        except Exception as exc:
            if verbose:
                print(f"  [warn] {qid}: {exc}", file=sys.stderr)
            pred = {
                "question_id": qid,
                "answer_value": None,
                "evidence_rows": [],
                "final_answer": "",
            }
        predictions.append(pred)
        if verbose and (i + 1) % 100 == 0:
            print(f"  DataQA: {i+1}/{len(questions)}", file=sys.stderr)
        time.sleep(_CALL_DELAY)
    _write_jsonl(output_dir / "pred_dataset2_dataqa.jsonl", predictions)
    return predictions


def evaluate_dataset3(
    config: ProviderConfig,
    output_dir: Path,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Run Dataset 3 (Writing) and return predictions with document text."""
    prompts = _read_jsonl(DATASET3 / "writing_prompts_public.jsonl")
    predictions: list[dict[str, Any]] = []
    for i, row in enumerate(prompts):
        qid = row["question_id"]
        messages = _build_writing_messages(row["prompt"])
        try:
            text = completion_text(messages, config)
            pred = {"question_id": qid, "answer": text.strip()}
        except Exception as exc:
            if verbose:
                print(f"  [warn] {qid}: {exc}", file=sys.stderr)
            pred = {"question_id": qid, "answer": ""}
        predictions.append(pred)
        if verbose and (i + 1) % 20 == 0:
            print(f"  Writing: {i+1}/{len(prompts)}", file=sys.stderr)
        time.sleep(_CALL_DELAY)
    _write_jsonl(output_dir / "pred_dataset3_writing.jsonl", predictions)
    return predictions


def evaluate_dataset4_audit(
    config: ProviderConfig,
    output_dir: Path,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Run Dataset 4 (Audit – find violations) and return predictions."""
    tasks = _read_jsonl(DATASET4 / "audit_tasks_public.jsonl")
    predictions: list[dict[str, Any]] = []
    for i, row in enumerate(tasks):
        qid = row["question_id"]
        messages = _build_audit_messages(row["prompt"])
        try:
            result = completion_json(messages, config)
            violations = result.get("violations", [])
            if not isinstance(violations, list):
                violations = []
            pred = {"question_id": qid, "violations": violations}
        except Exception as exc:
            if verbose:
                print(f"  [warn] {qid}: {exc}", file=sys.stderr)
            pred = {"question_id": qid, "violations": []}
        predictions.append(pred)
        if verbose and (i + 1) % 30 == 0:
            print(f"  Audit: {i+1}/{len(tasks)}", file=sys.stderr)
        time.sleep(_CALL_DELAY)
    _write_jsonl(output_dir / "pred_dataset4_audit.jsonl", predictions)
    return predictions


def evaluate_dataset4_rewrite(
    config: ProviderConfig,
    output_dir: Path,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Run Dataset 4 (Audit – rewrite/correct flawed document)."""
    tasks = _read_jsonl(DATASET4 / "audit_tasks_public.jsonl")
    predictions: list[dict[str, Any]] = []
    for i, row in enumerate(tasks):
        qid = row["question_id"]
        rewrite_prompt = row.get("rewrite_prompt", "")
        if not rewrite_prompt:
            continue
        messages = _build_writing_messages(rewrite_prompt)
        try:
            text = completion_text(messages, config)
            pred = {"question_id": qid, "rewrite": text.strip()}
        except Exception as exc:
            if verbose:
                print(f"  [warn] {qid}: {exc}", file=sys.stderr)
            pred = {"question_id": qid, "rewrite": ""}
        predictions.append(pred)
        if verbose and (i + 1) % 30 == 0:
            print(f"  Rewrite: {i+1}/{len(tasks)}", file=sys.stderr)
        time.sleep(_CALL_DELAY)
    _write_jsonl(output_dir / "pred_dataset4_rewrite.jsonl", predictions)
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


def _macro_score(scores: dict[str, Any]) -> float:
    """Average all leaf metric scores (0–1) across all datasets."""
    vals = []
    for dataset_scores in scores.values():
        if isinstance(dataset_scores, dict):
            for v in dataset_scores.values():
                if isinstance(v, float):
                    vals.append(v)
    return sum(vals) / len(vals) if vals else 0.0


def compile_leaderboard(output_dir: Path) -> dict[str, Any]:
    """Scan output_dir for per-model subdirs, score each, produce leaderboard.json."""
    entries = []
    for model_dir in sorted(output_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        meta_path = model_dir / "model_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"  Scoring {meta.get('name', model_dir.name)} …", file=sys.stderr)
        scores = _score_model(model_dir)
        macro = _macro_score(scores)
        entries.append({"model": meta, "scores": scores, "macro_avg": macro})

    entries.sort(key=lambda e: e["macro_avg"], reverse=True)
    for rank, entry in enumerate(entries, 1):
        entry["rank"] = rank

    # Bootstrap CI on macro_avg across models (treat each model's macro as one obs).
    if entries:
        macro_vals = [e["macro_avg"] for e in entries]
        ci_lo, ci_hi = _bootstrap_ci(macro_vals)
    else:
        ci_lo = ci_hi = 0.0

    board = {
        "leaderboard": entries,
        "bootstrap_ci_95": {"macro_avg_lo": ci_lo, "macro_avg_hi": ci_hi},
        "note": (
            "macro_avg is the unweighted mean of all leaf metric scores across "
            "all evaluated datasets. Scores marked N/A were not evaluated. "
            "95% CI is bootstrapped over model scores (n=1000 resamples)."
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


def run_model(
    spec: ModelSpec,
    datasets: list[str],
    base_output_dir: Path,
    verbose: bool = False,
) -> None:
    """Evaluate one model on the specified datasets."""
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
    (model_dir / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    do_all = "all" in datasets
    print(f"[{spec.name}] Starting evaluation …", file=sys.stderr)

    if do_all or "q" in datasets:
        print(f"  Dataset 1 (Q) …", file=sys.stderr)
        evaluate_dataset1(config, model_dir, verbose=verbose)

    if do_all or "dataqa" in datasets:
        print(f"  Dataset 2 (DataQA) …", file=sys.stderr)
        evaluate_dataset2(config, model_dir, verbose=verbose)

    if do_all or "writing" in datasets:
        print(f"  Dataset 3 (Writing) …", file=sys.stderr)
        evaluate_dataset3(config, model_dir, verbose=verbose)

    if do_all or "audit" in datasets:
        print(f"  Dataset 4 (Audit) …", file=sys.stderr)
        evaluate_dataset4_audit(config, model_dir, verbose=verbose)

    if do_all or "rewrite" in datasets:
        print(f"  Dataset 4 (Rewrite) …", file=sys.stderr)
        evaluate_dataset4_rewrite(config, model_dir, verbose=verbose)

    print(f"[{spec.name}] Done → {model_dir}", file=sys.stderr)


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
    for spec in specs:
        run_model(spec, datasets, args.output_dir, verbose=args.verbose)

    if len(specs) > 1:
        print("\nCompiling leaderboard …", file=sys.stderr)
        board = compile_leaderboard(args.output_dir)
        print(json.dumps(board, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
