from whisper_guard import GuardConfig, WhisperGuard, filter_hallucinations


def make_segment(text, no_speech_prob=0.1, avg_logprob=-0.5, compression_ratio=1.5,
                  start=None, end=None):
    seg = {
        "text": text,
        "no_speech_prob": no_speech_prob,
        "avg_logprob": avg_logprob,
        "compression_ratio": compression_ratio,
    }
    if start is not None:
        seg["start"] = start
    if end is not None:
        seg["end"] = end
    return seg


def test_normal_segments_pass():
    guard = WhisperGuard()
    result = guard.process([make_segment("hello"), make_segment("world")])
    assert result.passed is True
    assert result.text == "hello world"
    assert result.filtered_count == 2


def test_silence_rejection():
    guard = WhisperGuard()
    result = guard.process(
        [
            make_segment("noise", no_speech_prob=0.9),
            make_segment("still noise", no_speech_prob=0.85),
        ]
    )
    assert result.passed is False
    assert result.rejected_by == "silence"
    assert result.text == ""


def test_high_no_speech_filtered():
    guard = WhisperGuard()
    result = guard.process(
        [
            make_segment("keep this"),
            make_segment("drop this", no_speech_prob=0.95),
        ]
    )
    assert result.passed is True
    assert result.text == "keep this"
    assert result.filtered_count == 1


def test_low_logprob_filtered():
    guard = WhisperGuard()
    result = guard.process(
        [
            make_segment("keep this"),
            make_segment("drop this", avg_logprob=-2.0),
        ]
    )
    assert result.passed is True
    assert result.text == "keep this"
    assert result.filtered_count == 1


def test_compression_ratio_filtered():
    guard = WhisperGuard()
    result = guard.process(
        [
            make_segment("keep this"),
            make_segment("drop this", compression_ratio=3.5),
        ]
    )
    assert result.passed is True
    assert result.text == "keep this"
    assert result.filtered_count == 1


def test_repetitive_text_rejected():
    guard = WhisperGuard()
    repetitive = "abc123 " * 12
    result = guard.process([make_segment(repetitive)])
    assert result.passed is False
    assert result.rejected_by == "repetition"
    assert result.text == ""


def test_char_loops_removed():
    guard = WhisperGuard()
    result = guard.process([make_segment("ha ha hahaha xyzxyzxyz done")])
    assert result.passed is True
    assert "xyzxyzxyz" not in result.text
    assert result.char_loops_removed >= 1


def test_empty_segments():
    guard = WhisperGuard()
    result = guard.process([])
    assert result.passed is False
    assert result.rejected_by == "no_good_segments"
    assert result.text == ""


def test_custom_config():
    guard = WhisperGuard(GuardConfig(no_speech_prob=0.95))
    result = guard.process(
        [
            make_segment("keep this"),
            make_segment("also keep", no_speech_prob=0.9),
        ]
    )
    assert result.passed is True
    assert result.filtered_count == 2


def test_short_segment_stricter_logprob():
    """Short segments (<1.6s) use avg_logprob_short=-1.7 instead of -1.5."""
    guard = WhisperGuard()
    # logprob -1.6: passes normal threshold (-1.5) but fails short threshold (-1.7)
    result = guard.process([
        make_segment("good segment", start=0.0, end=5.0),
        make_segment("short hallucination", avg_logprob=-1.6, start=10.0, end=11.0),
    ])
    assert result.passed is True
    assert result.filtered_count == 2  # both kept — short one at -1.6 > -1.7


def test_short_segment_rejected_by_stricter_logprob():
    """Short segment with very low logprob gets rejected by stricter threshold."""
    guard = WhisperGuard()
    result = guard.process([
        make_segment("good segment", start=0.0, end=5.0),
        make_segment("bad short", avg_logprob=-1.8, start=10.0, end=11.0),
    ])
    assert result.passed is True
    assert result.text == "good segment"
    assert result.filtered_count == 1


def test_long_segment_uses_normal_logprob():
    """Long segments use the normal -1.5 threshold, not the stricter one."""
    guard = WhisperGuard()
    result = guard.process([
        make_segment("good segment", start=0.0, end=5.0),
        make_segment("borderline long", avg_logprob=-1.6, start=10.0, end=15.0),
    ])
    assert result.passed is True
    # -1.6 < -1.5 → filtered out by normal threshold
    assert result.text == "good segment"
    assert result.filtered_count == 1


def test_no_timing_uses_normal_logprob():
    """Segments without start/end fall back to normal threshold."""
    guard = WhisperGuard()
    result = guard.process([
        make_segment("good segment"),
        make_segment("no timing", avg_logprob=-1.6),
    ])
    assert result.passed is True
    # No timing info → uses normal -1.5 threshold → -1.6 < -1.5 → filtered
    assert result.text == "good segment"
    assert result.filtered_count == 1


def test_filter_hallucinations_convenience():
    filtered = filter_hallucinations(
        [
            make_segment("keep this"),
            make_segment("drop this", no_speech_prob=0.95),
        ]
    )
    assert len(filtered) == 1
    assert filtered[0]["text"] == "keep this"
