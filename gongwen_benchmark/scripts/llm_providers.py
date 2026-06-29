"""Unified LLM provider abstraction for CN-GongWen benchmark.

Three backends are supported:
  litellm — any LiteLLM-compatible provider (OpenAI, Anthropic, Minimax, Qwen, DeepSeek …)
  azure   — Azure OpenAI service (routed through LiteLLM's azure/ prefix)
  poe     — Poe API (via fastapi-poe; async client wrapped for sync use)

All backends share: SHA-256 disk cache, exponential-backoff retry, JSON extraction,
and the fact-guard that prevents LLM rewrites from altering key document tokens
(机关代字, 发文字号, 日期, 数值, 密级/紧急程度).

Environment variables (organised by provider):

  Common
    CN_GW_LLM_CACHE          cache directory (default: .cache/gongwen_llm)
    CN_GW_LLM_TEMPERATURE    default 0.35
    CN_GW_LLM_TIMEOUT        default 60 s (120 s for Poe)
    CN_GW_LLM_RETRIES        default 3

  LiteLLM (any OpenAI-compatible or natively supported provider)
    LLM_MODEL        model string, e.g. "openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022",
                     "qwen/qwen-max", "deepseek/deepseek-chat", "minimax/MiniMax-M1"
    LLM_API_KEY      provider API key
    LLM_API_BASE     custom base URL (leave unset for native providers)
    Fallbacks: OPENAI_API_KEY, OPENAI_API_BASE, MINIMAX_API_KEY, MINIMAX_API_BASE,
               ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, TOGETHER_API_KEY

  Azure OpenAI
    AZURE_DEPLOYMENT       deployment name, e.g. "gpt-4o"
    AZURE_API_KEY          Azure OpenAI key
    AZURE_API_BASE         https://<resource>.openai.azure.com/
    AZURE_API_VERSION      default "2024-08-01-preview"
    Aliases: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT

  Poe
    POE_API_KEY    Poe API key (from poe.com/api_key)
    POE_BOT_NAME   Poe bot name, e.g. "GPT-4o", "Claude-3-7-Sonnet", "DeepSeek-V3"
                   (case-sensitive, must match the bot's handle on poe.com)
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

CACHE_DIR = Path(os.getenv("CN_GW_LLM_CACHE", ".cache/gongwen_llm"))

# Fact tokens that must be preserved verbatim by any LLM rewrite.
FACT_PATTERN = re.compile(
    r"GA\d{3}|GW\d{2}|〔\d{4}〕\d+号|\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)?%?|绝密|机密|秘密|特急|加急"
)

# JSON fence stripper — some models wrap output in ```json … ``` blocks.
_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")


# ──────────────────────────────────────────────────────────────────────────────
# Provider configuration dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LiteLLMConfig:
    """Any LiteLLM-compatible provider.

    Model string format examples:
      "openai/gpt-4o"
      "openai/gpt-4o-mini"
      "anthropic/claude-3-5-sonnet-20241022"
      "anthropic/claude-opus-4-8"
      "qwen/qwen-max"
      "deepseek/deepseek-chat"
      "minimax/MiniMax-M1"          (via api_base relay)
      "together_ai/meta-llama/…"
      "MiniMax-M1"                  (legacy; auto-prefixed as openai/MiniMax-M1)
    """
    model: str = os.getenv("LLM_MODEL") or os.getenv("MINIMAX_MODEL", "openai/gpt-4o-mini")
    api_base: str | None = (
        os.getenv("LLM_API_BASE") or os.getenv("OPENAI_API_BASE") or os.getenv("MINIMAX_API_BASE")
    )
    api_key: str | None = (
        os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or
        os.getenv("MINIMAX_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or
        os.getenv("DEEPSEEK_API_KEY") or os.getenv("TOGETHER_API_KEY")
    )
    temperature: float = float(os.getenv("CN_GW_LLM_TEMPERATURE", "0.35"))
    timeout: int = int(os.getenv("CN_GW_LLM_TIMEOUT", "60"))
    retries: int = int(os.getenv("CN_GW_LLM_RETRIES", "3"))

    @property
    def litellm_model(self) -> str:
        known_prefixes = (
            "openai/", "anthropic/", "azure/", "qwen/", "deepseek/",
            "minimax/", "together_ai/", "bedrock/", "vertex_ai/",
            "groq/", "mistral/", "cohere/", "huggingface/", "ollama/",
        )
        if any(self.model.startswith(p) for p in known_prefixes):
            return self.model
        # Legacy bare names (e.g. "MiniMax-M1") → treat as OpenAI-compat relay.
        return f"openai/{self.model}"


@dataclass(frozen=True)
class AzureConfig:
    """Azure OpenAI service.

    LiteLLM routes azure/<deployment> calls to the Azure endpoint automatically.
    All Azure-specific parameters (api_version) are forwarded via extra_kwargs.
    """
    deployment: str = os.getenv("AZURE_DEPLOYMENT", "gpt-4o")
    api_base: str | None = (
        os.getenv("AZURE_API_BASE") or os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    api_key: str | None = (
        os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    )
    api_version: str = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
    temperature: float = float(os.getenv("CN_GW_LLM_TEMPERATURE", "0.35"))
    timeout: int = int(os.getenv("CN_GW_LLM_TIMEOUT", "60"))
    retries: int = int(os.getenv("CN_GW_LLM_RETRIES", "3"))

    @property
    def litellm_model(self) -> str:
        return f"azure/{self.deployment}"


@dataclass(frozen=True)
class PoeConfig:
    """Poe API via fastapi-poe.

    Bot names are the Poe handle (case-sensitive), e.g.:
      "GPT-4o", "GPT-4o-Mini", "Claude-3-7-Sonnet", "Claude-3-5-Sonnet",
      "DeepSeek-V3", "Gemini-2.0-Flash", "Qwen2.5-72B-Instruct"

    Install: pip install 'fastapi-poe>=0.0.47'
    Get API key: https://poe.com/api_key
    """
    bot_name: str = os.getenv("POE_BOT_NAME", "GPT-4o-Mini")
    api_key: str | None = os.getenv("POE_API_KEY")
    temperature: float = float(os.getenv("CN_GW_LLM_TEMPERATURE", "0.35"))
    timeout: int = int(os.getenv("CN_GW_LLM_TIMEOUT", "120"))
    retries: int = int(os.getenv("CN_GW_LLM_RETRIES", "3"))


ProviderConfig = Union[LiteLLMConfig, AzureConfig, PoeConfig]


# ──────────────────────────────────────────────────────────────────────────────
# Availability checks
# ──────────────────────────────────────────────────────────────────────────────

def litellm_available() -> bool:
    return importlib.util.find_spec("litellm") is not None


def poe_available() -> bool:
    return importlib.util.find_spec("fastapi_poe") is not None


# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cache_key(messages: list[dict[str, str]], config: ProviderConfig) -> str:
    if isinstance(config, AzureConfig):
        ident = f"azure::{config.deployment}::{config.api_base}"
    elif isinstance(config, PoeConfig):
        ident = f"poe::{config.bot_name}"
    else:
        ident = f"litellm::{config.litellm_model}::{config.api_base}"
    payload = json.dumps(
        {"provider": ident, "messages": messages},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# JSON extraction helper
# ──────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from raw model text (handles fenced blocks)."""
    # Strip markdown fences first.
    m = _JSON_FENCE.search(text)
    candidate = m.group(1) if m else text
    # Find the first {...} span.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1:
        candidate = candidate[start : end + 1]
    result = json.loads(candidate)
    if not isinstance(result, dict):
        raise ValueError("LLM response must be a JSON object")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# LiteLLM backend (also handles Azure via azure/ prefix)
# ──────────────────────────────────────────────────────────────────────────────

def _litellm_call(
    messages: list[dict[str, str]],
    config: LiteLLMConfig | AzureConfig,
    json_mode: bool = True,
) -> str:
    if not litellm_available():
        raise RuntimeError(
            "litellm is not installed. Run: pip install 'cn-gongwen-benchmark[llm]'"
        )
    api_key = config.api_key
    if not api_key:
        raise RuntimeError(
            f"No API key found for provider {type(config).__name__}. "
            "Set LLM_API_KEY, AZURE_API_KEY, or the relevant provider env var."
        )
    litellm = importlib.import_module("litellm")
    kwargs: dict[str, Any] = dict(
        model=config.litellm_model,
        messages=messages,
        temperature=config.temperature,
        timeout=config.timeout,
    )
    if config.api_base:
        kwargs["api_base"] = config.api_base
    kwargs["api_key"] = api_key
    if isinstance(config, AzureConfig):
        kwargs["api_version"] = config.api_version
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error: Exception | None = None
    for attempt in range(config.retries):
        try:
            response = litellm.completion(**kwargs)
            return response["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"LiteLLM call failed after {config.retries} attempts: {last_error}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Poe backend
# ──────────────────────────────────────────────────────────────────────────────

def _poe_call(messages: list[dict[str, str]], config: PoeConfig) -> str:
    if not poe_available():
        raise RuntimeError(
            "fastapi-poe is not installed. Run: pip install 'cn-gongwen-benchmark[poe]'"
        )
    if not config.api_key:
        raise RuntimeError(
            "POE_API_KEY is required for Poe provider. "
            "Get yours at https://poe.com/api_key"
        )
    fp = importlib.import_module("fastapi_poe")

    async def _async_call() -> str:
        poe_messages = [
            fp.ProtocolMessage(role=m["role"], content=m["content"])
            for m in messages
        ]
        text = ""
        async for chunk in fp.get_bot_response(
            messages=poe_messages,
            bot_name=config.bot_name,
            api_key=config.api_key,
        ):
            if hasattr(chunk, "text"):
                text += chunk.text
        return text

    last_error: Exception | None = None
    for attempt in range(config.retries):
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(asyncio.run, _async_call())
                        return future.result(timeout=config.timeout)
            except RuntimeError:
                pass
            return asyncio.run(_async_call())
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Poe call failed after {config.retries} attempts: {last_error}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def completion_text(
    messages: list[dict[str, str]],
    config: ProviderConfig | None = None,
) -> str:
    """Return raw text response from the model (no JSON parsing, no caching).

    Use for Dataset-3 writing tasks where output is free-form document text.
    Caching is skipped because writing output length varies and cache benefit is
    lower; callers that want caching should wrap this themselves.
    """
    cfg = config or LiteLLMConfig()
    if isinstance(cfg, PoeConfig):
        return _poe_call(messages, cfg)
    return _litellm_call(messages, cfg, json_mode=False)


def completion_json(
    messages: list[dict[str, str]],
    config: ProviderConfig | None = None,
) -> dict[str, Any]:
    """Return a parsed JSON dict from the model, with SHA-256 disk caching and retry.

    Raises RuntimeError after all retries are exhausted.
    """
    cfg = config or LiteLLMConfig()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{_cache_key(messages, cfg)}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    last_error: Exception | None = None
    retries = cfg.retries
    for attempt in range(retries):
        try:
            if isinstance(cfg, PoeConfig):
                raw = _poe_call(messages, cfg)
                result = _extract_json(raw)
            else:
                try:
                    raw = _litellm_call(messages, cfg, json_mode=True)
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    # Some providers ignore response_format; fall back to extraction.
                    result = _extract_json(raw)
            if not isinstance(result, dict):
                raise ValueError("LLM response must be a JSON object")
            cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            return result
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"completion_json failed after {retries} attempts: {last_error}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fact-guard (shared with generation pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def facts(text: str) -> set[str]:
    return set(FACT_PATTERN.findall(text))


def fact_guard(source: str, candidate: str) -> bool:
    """Return True only when the candidate preserves exactly the source's fact tokens."""
    return facts(source) == facts(candidate) and bool(candidate.strip())


# ──────────────────────────────────────────────────────────────────────────────
# Generation-time helpers (used by generate_benchmarks.py and writing generator)
# ──────────────────────────────────────────────────────────────────────────────

def rewrite_question(
    template_question: str,
    context: dict[str, Any],
    config: ProviderConfig | None = None,
) -> str:
    """Rewrite a Q-dataset question with a fresh surface form while preserving all facts.

    Falls back to the original template on any error (network / key / JSON / fact violation)
    so the frozen generation pipeline is never interrupted.
    """
    payload = {
        "template_question": template_question,
        "context": context,
        "strict_constraints": [
            "保持文种、行文方向、发文字号、机关代字、日期、密级、紧急程度和安全意图不变",
            "不得新增数值、法规、机关、个人或密级信息",
            '输出 {"question": "..."}',
        ],
    }
    try:
        result = completion_json(
            [
                {"role": "system", "content": "你是党政机关公文写作与办理评测问题改写助手。"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            config,
        )
    except Exception:
        return template_question
    candidate = str(result.get("question", "")).strip()
    return candidate if fact_guard(template_question, candidate) else template_question


def polish_briefing(
    template_answer: str,
    evidence: list[dict[str, Any]],
    config: ProviderConfig | None = None,
) -> str:
    """Polish a DataQA briefing answer while preserving all numerical facts.

    Falls back to the deterministic template on any error.
    """
    payload = {
        "template_answer": template_answer,
        "evidence": evidence,
        "strict_constraints": [
            "只能使用 evidence 中的事实",
            "不得增加文种、机关、字号、日期、密级或整改结论",
            '输出 {"answer": "..."}',
        ],
    }
    try:
        result = completion_json(
            [
                {"role": "system", "content": "你是审慎的公文办理播报润色助手。"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            config,
        )
    except Exception:
        return template_answer
    candidate = str(result.get("answer", "")).strip()
    return candidate if fact_guard(template_answer, candidate) else template_answer


# ──────────────────────────────────────────────────────────────────────────────
# Config factory — build from CLI / env strings
# ──────────────────────────────────────────────────────────────────────────────

def config_from_provider(provider: str, model: str | None = None) -> ProviderConfig:
    """Build the appropriate config dataclass from a provider name string.

    provider values: "litellm", "azure", "poe"
    model: optional override for model/deployment/bot_name.
    """
    p = provider.lower().strip()
    if p == "azure":
        return AzureConfig(
            deployment=model or os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
        )
    if p == "poe":
        return PoeConfig(
            bot_name=model or os.getenv("POE_BOT_NAME", "GPT-4o-Mini"),
        )
    # Default: litellm
    return LiteLLMConfig(
        model=model or os.getenv("LLM_MODEL") or os.getenv("MINIMAX_MODEL", "openai/gpt-4o-mini"),
    )
