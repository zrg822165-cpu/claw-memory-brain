"""Lightweight hard filter for candidate event chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .event_chunker import EventChunk

_NON_WORD_RE = re.compile(r"[\s`*_>#\-\.\,\!\?;:'\"\(\)\[\]\{\}，。！？；：“”‘’、/\\]+")
_TECH_SIGNAL_RE = re.compile(r"[A-Za-z0-9_/\\:\.-]|[\u4e00-\u9fff]")

_CHATTER_VALUES = {
    "好",
    "好的",
    "收到",
    "明白",
    "知道了",
    "嗯",
    "嗯嗯",
    "哈哈",
    "行",
    "可以",
    "ok",
    "okay",
    "yes",
    "no",
}


@dataclass
class FilterDecision:
    keep: bool
    reason: str = ""


def _normalize_for_filter(text: str) -> str:
    lowered = text.strip().lower()
    return _NON_WORD_RE.sub("", lowered)


def hard_filter_candidate(chunk: EventChunk) -> FilterDecision:
    """最小兜底过滤：
    - 空 content / 空 summary
    - 明确闲聊短句
    - 明显没有任何信息量的坏块
    """
    content = chunk.content.strip()
    summary = chunk.chunk_summary.strip()
    if not content:
        return FilterDecision(False, "empty_content")
    if not summary:
        return FilterDecision(False, "empty_summary")

    normalized_content = _normalize_for_filter(content)
    normalized_summary = _normalize_for_filter(summary)
    if not normalized_content:
        return FilterDecision(False, "blank_like_content")
    if not normalized_summary:
        return FilterDecision(False, "blank_like_summary")

    # 只拦最明显的闲聊短句，避免误伤真实单句经验。
    if normalized_content in _CHATTER_VALUES and len(normalized_content) <= 6:
        return FilterDecision(False, "plain_chatter")

    # 极短且没有任何技术/语义承载的块直接过滤。
    if len(normalized_content) < 4 and not _TECH_SIGNAL_RE.search(content):
        return FilterDecision(False, "too_short_low_signal")

    return FilterDecision(True, "")
