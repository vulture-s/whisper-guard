import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GuardConfig:
    silence_threshold: float = 0.6
    no_speech_prob: float = 0.8
    avg_logprob: float = -1.5
    avg_logprob_short: float = -1.7
    short_segment_threshold: float = 1.6
    compression_ratio: float = 3.0
    repetition_window: int = 6
    repetition_threshold: float = 0.35
    char_loop_min_pattern: int = 2
    char_loop_max_pattern: int = 4
    char_loop_min_repeats: int = 3


@dataclass
class GuardResult:
    text: str
    passed: bool
    rejected_by: Optional[str] = None
    original_count: int = 0
    filtered_count: int = 0
    char_loops_removed: int = 0


class WhisperGuard:
    def __init__(self, config: Optional[GuardConfig] = None):
        self.config = config or GuardConfig()
        self._loop_pattern = self._compile_char_loop_pattern()

    def process(self, segments: List[Dict]) -> GuardResult:
        if not segments:
            return GuardResult(
                text="",
                passed=False,
                rejected_by="no_good_segments",
                original_count=0,
                filtered_count=0,
            )

        avg_no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
        if avg_no_speech > self.config.silence_threshold:
            return GuardResult(
                text="",
                passed=False,
                rejected_by="silence",
                original_count=len(segments),
                filtered_count=0,
            )

        good_segments = self._filter_segments(segments)

        if not good_segments:
            return GuardResult(
                text="",
                passed=False,
                rejected_by="no_good_segments",
                original_count=len(segments),
                filtered_count=0,
            )

        filtered_text = " ".join(segment["text"] for segment in good_segments).strip()
        if self.is_repetitive(filtered_text):
            return GuardResult(
                text="",
                passed=False,
                rejected_by="repetition",
                original_count=len(segments),
                filtered_count=len(good_segments),
            )

        cleaned_text, removed = self.remove_char_loops(filtered_text)
        return GuardResult(
            text=cleaned_text,
            passed=bool(cleaned_text),
            original_count=len(segments),
            filtered_count=len(good_segments),
            char_loops_removed=removed,
        )

    def is_repetitive(self, text: str) -> bool:
        window = self.config.repetition_window
        if len(text) < window * 3:
            words = [word for word in text.split() if word]
            if len(words) < 3:
                return False
            return (len(set(words)) / len(words)) < self.config.repetition_threshold

        words = [word for word in text.split() if word]
        if len(words) >= 3:
            word_ratio = len(set(words)) / len(words)
            if word_ratio < self.config.repetition_threshold:
                return True

        chunks = [text[i:i + window] for i in range(0, len(text) - window, window)]
        if not chunks:
            return False
        unique = len(set(chunks))
        return (unique / len(chunks)) < self.config.repetition_threshold

    def has_char_loops(self, text: str) -> bool:
        return bool(self._loop_pattern.search(text))

    def remove_char_loops(self, text: str) -> tuple:
        cleaned, count = self._loop_pattern.subn(r"\1", text)
        return cleaned, count

    def _filter_segments(self, segments: List[Dict]) -> List[Dict]:
        good = []
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue
            if segment.get("no_speech_prob", 0) > self.config.no_speech_prob:
                continue
            duration = None
            if "start" in segment and "end" in segment:
                duration = segment["end"] - segment["start"]
            if duration is not None and duration < self.config.short_segment_threshold:
                logprob_thresh = self.config.avg_logprob_short
            else:
                logprob_thresh = self.config.avg_logprob
            if segment.get("avg_logprob", 0) < logprob_thresh:
                continue
            if segment.get("compression_ratio", 1) > self.config.compression_ratio:
                continue
            cleaned = dict(segment)
            cleaned["text"] = text
            good.append(cleaned)
        return good

    def _compile_char_loop_pattern(self):
        min_pattern = self.config.char_loop_min_pattern
        max_pattern = self.config.char_loop_max_pattern
        min_repeats = self.config.char_loop_min_repeats
        return re.compile(r"(.{%d,%d})\1{%d,}" % (min_pattern, max_pattern, min_repeats - 1))


def filter_hallucinations(segments: List[Dict], config: Optional[GuardConfig] = None) -> List[Dict]:
    guard = WhisperGuard(config)
    result = guard.process(segments)
    if not result.passed:
        return []

    filtered = []
    for seg in guard._filter_segments(segments):
        seg["text"] = guard._loop_pattern.sub(r"\1", seg["text"]).strip()
        if seg["text"]:
            filtered.append(seg)
    return filtered
