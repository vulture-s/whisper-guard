import math
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
    # L3 word-ratio is measured per window of this many words and averaged.
    # A whole-text type/token ratio falls with length for any natural text, so
    # without a window every long genuine transcript eventually reads as a loop.
    repetition_word_window: int = 100
    char_loop_min_pattern: int = 2
    char_loop_max_pattern: int = 4
    char_loop_min_repeats: int = 3
    # A loop whose unit is purely numeric ("00", "12", ".000") is data unless
    # the repeated run is at least this long: amounts, phone numbers, IPs and
    # versions stay intact, while a decoder spewing "0000…" still collapses.
    char_loop_numeric_min_span: int = 20
    # L2 no_speech is not trusted alone once timing is known. Over music or a
    # bed of noise faster-whisper can report no_speech_prob ~1.0 on a 30 s
    # window of real, confidently decoded speech (lp -0.02, 7 chars/s) and the
    # filter silently dropped it. What separates the two is density: silence
    # hallucinations are one short phrase stretched over a long window
    # ("Thank you." over 30 s = 0.3 chars/s), real speech is 2–15 chars/s.
    # So a high-no_speech segment is kept only when it is dense AND confident.
    # avg_logprob is no help on its own: measured hallucinations sat at -0.04
    # to -0.24, so Whisper's own "no_speech high AND logprob < -1" keeps them.
    no_speech_rescue_density: float = 2.0
    no_speech_rescue_logprob: float = -0.5
    # Judged per decoding window, not per segment: faster-whisper gives every
    # segment of one 30 s window the same no_speech/avg_logprob, so consecutive
    # segments sharing both are pooled. A window shorter than this is never
    # rescued — a 1 s "謝謝觀看" over silence is dense by construction.
    no_speech_rescue_min_duration: float = 5.0
    # Long + sparse is a hallucination whatever no_speech says. Measured silence
    # / noise hallucinations ("Thank you." over 30 s) came with no_speech 0.14,
    # 0.75, 0.80 and 0.87 and avg_logprob -0.04…-0.24, so neither metric can
    # catch the low-no_speech ones; density can: every hallucinated window
    # ≥10 s was ≤0.53 chars/s, every real-speech window ≥4.9 chars/s
    # (faster-whisper large-v3, 2026-10-19 probe). Same window grouping and
    # char counting as the rescue rule.
    sparse_min_duration: float = 10.0
    sparse_density: float = 1.0


@dataclass
class GuardResult:
    text: str
    passed: bool
    rejected_by: Optional[str] = None
    original_count: int = 0
    filtered_count: int = 0
    char_loops_removed: int = 0


# Numeric "data" units: digits and in-number punctuation, NO whitespace —
# "0 0 0 0 0 0" is a decoder loop, not a value (Codex audit of #1).
_NUMERIC_UNIT = re.compile(r"^[\d.,:/+\-$%]+$")
# A thousands group (",000" / ".000") is always data, however many repeat:
# "$1,000,000,000,000,000" must not become "$1,000".
_THOUSANDS_GROUP = re.compile(r"^[.,]\d{3}$")


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

        # Dense, confident segments are speech whatever no_speech says (see
        # GuardConfig.no_speech_rescue_*): a 60 s reel over BGM that faster-whisper
        # decodes as two 30 s windows at no_speech ~1.0 was wiped here, before L2.
        rescued = self._rescue_flags(segments)
        l1_view = [dict(s, no_speech_prob=0.0) if r else s for s, r in zip(segments, rescued)]
        if self._mean_no_speech(l1_view) > self.config.silence_threshold:
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
        """Batch-level no_speech score for the silence gate (L1).

        The gate drops the WHOLE batch, so it must only fire when the batch is
        silent by both measures:

        * by segment count alone, three 1 s BGM tails outvoted 30 s of clear
          speech and wiped the transcript;
        * by duration alone, 40 s of interview followed by 80 s of ambience
          (three long hallucinated segments) wiped the interview — speech the
          count-based gate kept (dual-track audit of #1).

        So: min(count mean, duration-weighted mean). It never rejects a batch
        the old count gate passed; long silent segments are still dropped one
        by one by the per-segment no_speech filter. Without usable timing
        (any segment lacking a positive start/end) it is the plain mean.
        """
        durations = []
        for segment in segments:
            if "start" not in segment or "end" not in segment:
                durations = None
                break
            start, end = segment["start"], segment["end"]
            # Only real numbers: a numeric *string* used to be float()-ed here,
            # pass L1, then crash in _filter_segments ("30" - "0"), where the
            # plain mean had rejected the batch before ever getting there.
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       and math.isfinite(v) for v in (start, end)):
                durations = None
                break
            duration = float(end) - float(start)
            if duration <= 0:
                durations = None
                break
            durations.append(duration)

        probs = [s.get("no_speech_prob", 0) for s in segments]
        plain = sum(probs) / len(probs)
        if durations:
            total = sum(durations)
            return min(plain, sum(p * d for p, d in zip(probs, durations)) / total)
        return plain

    def is_repetitive(self, text: str) -> bool:
        window = self.config.repetition_window
        if len(text) < window * 3:
            words = [word for word in text.split() if word]
            if len(words) < 3:
                return False
            return (len(set(words)) / len(words)) < self.config.repetition_threshold

        words = [word for word in text.split() if word]
        if len(words) >= 3:
            if self._word_ratio(words) < self.config.repetition_threshold:
                return True

        chunks = [text[i:i + window] for i in range(0, len(text) - window, window)]
        if not chunks:
            return False
        unique = len(set(chunks))
        return (unique / len(chunks)) < self.config.repetition_threshold

    def _word_ratio(self, words: List[str]) -> float:
        """Unique-word ratio, averaged over fixed windows for long texts.

        Up to one window this is the plain ratio (unchanged behaviour). Beyond
        it, each full window is scored on its own and the scores are averaged;
        the trailing partial window is folded into the last full one. A loop
        drives its windows to ~0 wherever it sits, while real speech keeps
        every window high no matter how long the recording is.
        """
        window = max(int(self.config.repetition_word_window), 3)
        if len(words) <= window:
            return len(set(words)) / len(words)
        chunks = [words[i:i + window] for i in range(0, len(words), window)]
        if len(chunks) > 1 and len(chunks[-1]) < window:
            tail = chunks.pop()
            chunks[-1] = chunks[-1] + tail
        return sum(len(set(c)) / len(c) for c in chunks) / len(chunks)

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
        if not any(ch.isdecimal() for ch in unit):
            return False
        if not _NUMERIC_UNIT.match(unit):
            return False
        if _THOUSANDS_GROUP.match(unit):
            return True
        return len(match.group(0)) < self.config.char_loop_numeric_min_span

    @staticmethod
    def _duration(segment: Dict) -> Optional[float]:
        start, end = segment.get("start"), segment.get("end")
        for value in (start, end):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return None
        duration = float(end) - float(start)
        return duration if duration > 0 else None

    @staticmethod
    def _density(text: str, duration: float) -> float:
        # Letters/digits/CJK per second; spaces and punctuation don't count.
        return len(re.sub(r"[\W_]", "", text)) / duration

    def _windows(self, segments: List[Dict]):
        """Yield (i, j, span) per decoding window segments[i:j].

        Consecutive segments with identical (no_speech_prob, avg_logprob) are one
        faster-whisper decoding window. span is None when any segment in it lacks
        usable timing.
        """
        i = 0
        while i < len(segments):
            key = (segments[i].get("no_speech_prob"), segments[i].get("avg_logprob"))
            j = i + 1
            while j < len(segments) and (segments[j].get("no_speech_prob"),
                                         segments[j].get("avg_logprob")) == key:
                j += 1
            window = segments[i:j]
            span = None
            if all(self._duration(s) is not None for s in window):
                span = float(window[-1]["end"]) - float(window[0]["start"])
            yield i, j, span
            i = j

    @staticmethod
    def _window_text(segments: List[Dict]) -> str:
        return "".join(s.get("text", "") for s in segments)

    def _rescue_flags(self, segments: List[Dict]) -> List[bool]:
        """Per segment: is it speech despite a high no_speech_prob?

        The decoding window is rescued when it is long enough, dense and
        confident. Any segment without usable timing is never rescued.
        """
        flags = [False] * len(segments)
        for i, j, span in self._windows(segments):
            if span is None:
                continue
            logprob = segments[i].get("avg_logprob")
            ok = (
                span >= self.config.no_speech_rescue_min_duration
                and self._density(self._window_text(segments[i:j]), span) >= self.config.no_speech_rescue_density
                and (logprob if logprob is not None else 0) > self.config.no_speech_rescue_logprob
            )
            if ok:
                flags[i:j] = [True] * (j - i)
        return flags

    def _sparse_flags(self, segments: List[Dict]) -> List[bool]:
        """Per segment: is its decoding window long and nearly empty?

        A short phrase stretched over a long window is what Whisper emits over
        silence or noise. Windows without usable timing are never flagged.
        """
        flags = [False] * len(segments)
        for i, j, span in self._windows(segments):
            if span is None or span < self.config.sparse_min_duration:
                continue
            if self._density(self._window_text(segments[i:j]), span) < self.config.sparse_density:
                flags[i:j] = [True] * (j - i)
        return flags

    def _filter_segments(self, segments: List[Dict]) -> List[Dict]:
        good = []
        rescued = self._rescue_flags(segments)
        sparse = self._sparse_flags(segments)
        for segment, is_rescued, is_sparse in zip(segments, rescued, sparse):
            text = segment.get("text", "").strip()
            if not text or is_sparse:
                continue
            if segment.get("no_speech_prob", 0) > self.config.no_speech_prob and not is_rescued:
                continue
            duration = self._duration(segment)
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
