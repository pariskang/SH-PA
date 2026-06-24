"""Production LiteLLM helpers for Minimax via an OpenAI-compatible relay.

LLM 输出经过磁盘缓存、指数退避重试、JSON 校验，并用事实护栏防止改写时引入或删除
文种、行文方向、发文字号、机关代字、日期、密级等关键事实。任何公文要素抽取、格式合规
判定、排序与统计标签都不依赖 LLM，全部由 Python 从 records.csv 确定性计算。
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_DIR = Path(os.getenv("CN_GW_LLM_CACHE", ".cache/gongwen_litellm"))
# 关键事实：机关/文书代码、发文字号、日期、百分比/数值、密级与紧急程度词
FACT_PATTERN = re.compile(
    r"GA\d{3}|GW\d{2}|〔\d{4}〕\d+号|\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)?%?|绝密|机密|秘密|特急|加急"
)


@dataclass(frozen=True)
class LiteLLMConfig:
    model: str = os.getenv("MINIMAX_MODEL", "MiniMax-M1")
    api_base: str | None = os.getenv("MINIMAX_API_BASE") or os.getenv("OPENAI_API_BASE")
    api_key: str | None = os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY")
    temperature: float = float(os.getenv("CN_GW_LLM_TEMPERATURE", "0.35"))
    timeout: int = int(os.getenv("CN_GW_LLM_TIMEOUT", "60"))
    retries: int = int(os.getenv("CN_GW_LLM_RETRIES", "3"))

    @property
    def litellm_model(self) -> str:
        return self.model if self.model.startswith(("openai/", "minimax/")) else f"openai/{self.model}"


def litellm_available() -> bool:
    return importlib.util.find_spec("litellm") is not None


def _cache_key(messages: list[dict[str, str]], config: LiteLLMConfig) -> str:
    payload = json.dumps({"model": config.litellm_model, "base": config.api_base, "messages": messages}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def completion_json(messages: list[dict[str, str]], config: LiteLLMConfig | None = None) -> dict[str, Any]:
    """Call LiteLLM with disk cache, exponential retry, and JSON validation."""
    cfg = config or LiteLLMConfig()
    if not cfg.api_key:
        raise RuntimeError("MINIMAX_API_KEY or OPENAI_API_KEY is required for LiteLLM calls.")
    if not litellm_available():
        raise RuntimeError("litellm is not installed. Install optional dependency: pip install '.[llm]'")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{_cache_key(messages, cfg)}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    litellm = importlib.import_module("litellm")
    last_error: Exception | None = None
    for attempt in range(cfg.retries):
        try:
            response = litellm.completion(model=cfg.litellm_model, api_base=cfg.api_base, api_key=cfg.api_key, messages=messages, temperature=cfg.temperature, timeout=cfg.timeout, response_format={"type": "json_object"})
            result = json.loads(response["choices"][0]["message"]["content"])
            if not isinstance(result, dict):
                raise ValueError("LLM response must be a JSON object")
            cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            return result
        except Exception as exc:  # network/provider/JSON failures are retried together
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"LiteLLM call failed after {cfg.retries} attempts: {last_error}")


def facts(text: str) -> set[str]:
    return set(FACT_PATTERN.findall(text))


def fact_guard(source: str, candidate: str) -> bool:
    """Reject rewritten text that introduces or removes explicit factual tokens."""
    return facts(source) == facts(candidate) and bool(candidate.strip())


def rewrite_question(template_question: str, context: dict[str, Any], config: LiteLLMConfig | None = None) -> str:
    payload = {"template_question": template_question, "context": context, "strict_constraints": ["保持文种、行文方向、发文字号、机关代字、日期、密级、紧急程度和安全意图不变", "不得新增数值、法规、机关、个人或密级信息", "输出 {\"question\": \"...\"}"]}
    try:  # 单条改写失败（网络/服务/JSON/无密钥）即回退确定性模板，保证冻结流程不中断
        result = completion_json([{"role": "system", "content": "你是党政机关公文写作与办理评测问题改写助手。"}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], config)
    except Exception:
        return template_question
    candidate = str(result.get("question", "")).strip()
    return candidate if fact_guard(template_question, candidate) else template_question


def polish_briefing(template_answer: str, evidence: list[dict[str, Any]], config: LiteLLMConfig | None = None) -> str:
    payload = {"template_answer": template_answer, "evidence": evidence, "strict_constraints": ["只能使用 evidence 中的事实", "不得增加文种、机关、字号、日期、密级或整改结论", "输出 {\"answer\": \"...\"}"]}
    try:  # 单条润色失败即回退确定性模板，保证冻结流程不中断
        result = completion_json([{"role": "system", "content": "你是审慎的公文办理播报润色助手。"}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], config)
    except Exception:
        return template_answer
    candidate = str(result.get("answer", "")).strip()
    return candidate if fact_guard(template_answer, candidate) else template_answer
