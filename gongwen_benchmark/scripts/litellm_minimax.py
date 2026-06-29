"""Backward-compatibility shim for litellm_minimax.

All implementation has moved to llm_providers.py which supports Poe, Azure,
and any LiteLLM-compatible provider in addition to Minimax.

Existing callers (generate_benchmarks.py, generate_writing_prompts.py, tests)
import LiteLLMConfig, completion_json, rewrite_question, polish_briefing, and
litellm_available from this module — those names are re-exported unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the scripts directory is on sys.path so that llm_providers is importable
# both when this file is run directly and when imported as gongwen_benchmark.scripts.litellm_minimax.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from llm_providers import (  # noqa: F401  (re-export for backward compat)
    CACHE_DIR,
    FACT_PATTERN,
    AzureConfig,
    LiteLLMConfig,
    PoeConfig,
    completion_json,
    completion_text,
    fact_guard,
    facts,
    litellm_available,
    poe_available,
    polish_briefing,
    rewrite_question,
)

__all__ = [
    "CACHE_DIR",
    "FACT_PATTERN",
    "AzureConfig",
    "LiteLLMConfig",
    "PoeConfig",
    "completion_json",
    "completion_text",
    "fact_guard",
    "facts",
    "litellm_available",
    "poe_available",
    "polish_briefing",
    "rewrite_question",
]
