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
    # A loop whose unit is purely numeric ("00", "12", ".000") is data unless
    # the repeated run is at least this long: amounts, phone numbers, IPs and
    # versions stay intact, while a decoder spewing "0000…" still collapses.
    char_loop_numeric_min_span: int = 20


@dataclass
class GuardResult:
    text: str
    passed: bool
    rejected_by: Optional[str] = None
    original_count: int = 0
    filtered_count: int = 0
    char_loops_removed: int = 0


_NUMERIC_UNIT = re.compile(r"^[\d\s.,:/+\-$%]+$")


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

        if self._mean_no_speech(segments) > self.config.silence_threshold:
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

    @staticmethod
    def _mean_no_speech(segments: List[Dict]) -> float:
        """Duration-weighted mean of no_speech_prob over the batch.

        Weighting by segment *count* let three 1 s BGM tails outvote 30 s of
        clear speech and wipe the whole transcript. When every segment has a
        positive duration (start/end), weight by it; otherwise fall back to the
        plain mean, which is the only thing the data supports.
        """
        durations = []
        for segment in segments:
            if "start" not in segment or "end" not in segment:
                durations = None
                break
            try:
                duration = float(segment["end"]) - float(segment["start"])
            except (TypeError, ValueError):
                durations = None
                break
            if duration <= 0:
                durations = None
                break
            durations.append(duration)

        probs = [s.get("no_speech_prob", 0) for s in segments]
        if durations:
            total = sum(durations)
            return sum(p * d for p, d in zip(probs, durations)) / total
        return sum(probs) / len(probs)

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
        return any(not self._is_numeric_data(m) for m in self._loop_pattern.finditer(text))

    def remove_char_loops(self, text: str) -> tuple:
        count = 0

        def _collapse(match):
            nonlocal count
            if self._is_numeric_data(match):
                return match.group(0)
            count += 1
            return match.group(1)

        cleaned = self._loop_pattern.sub(_collapse, text)
        return cleaned, count

    def _is_numeric_data(self, match) -> bool:
        """True when a matched loop is a number, not a decoder loop.

        Only a unit made of digits and numeric punctuation counts as data
        ("00" in 100000000, "12" in 0912121212, ".000" in 1.000.000.000).
        A unit that mixes a digit with text ("第1集", "ha1") is still a loop,
        and a purely numeric run of ``char_loop_numeric_min_span`` chars or
        more ("0000…" x32) is a decoder loop, not a value anyone said.
        """
        unit = match.group(1)
        if not any(ch.isdigit() for ch in unit):
            return False
        if not _NUMERIC_UNIT.match(unit):
            return False
        return len(match.group(0)) < self.config.char_loop_numeric_min_span

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
        # Numeric runs ("00" x4 inside 100000000) are exempted in
        # _is_numeric_data, not here: excluding every digit from the unit also
        # let digit-bearing hallucinations ("第1集第1集第1集", "0000…" x32)
        # through untouched.
        return re.compile(r"(.{%d,%d})\1{%d,}" % (min_pattern, max_pattern, min_repeats - 1))


def filter_hallucinations(segments: List[Dict], config: Optional[GuardConfig] = None) -> List[Dict]:
    guard = WhisperGuard(config)
    result = guard.process(segments)
    if not result.passed:
        return []

    filtered = []
    for seg in guard._filter_segments(segments):
        seg["text"] = guard.remove_char_loops(seg["text"])[0].strip()
        if seg["text"]:
            filtered.append(seg)
    return filtered
