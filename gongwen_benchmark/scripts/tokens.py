"""确定性 token 估算 + 可选 tiktoken 精确计数。

为保证基准“零随机、逐字节复现”，数据生成 / 校验 / 打分一律使用确定性估算器
``estimate_tokens``（CJK 每字≈1 token，其余每 4 字符≈1 token），因此长度分桶与
打分在任何环境都一致。评测方如需更贴近某具体 tokenizer 的精确值，可调用
``count_tokens``（装了 tiktoken 时启用，否则回退到同一估算器）。
"""
from __future__ import annotations

import re

# CJK 统一表意文字 + 扩展A + 兼容表意 + CJK 标点 + 全角符号（各计 1 token）
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿　-〿＀-￯]")


def estimate_tokens(text: str) -> int:
    """确定性 token 估算：CJK 字符各计 1，其余每 4 字符计 1（向上取整）。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def count_tokens(text: str, encoding: str = "o200k_base") -> int:
    """评测期可选的精确计数：装了 tiktoken 用其编码，否则回退到确定性估算器。"""
    try:
        import tiktoken  # type: ignore

        return len(tiktoken.get_encoding(encoding).encode(text))
    except Exception:
        return estimate_tokens(text)
