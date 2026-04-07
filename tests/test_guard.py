from whisper_guard import GuardConfig, WhisperGuard, filter_hallucinations


def make_segment(text, no_speech_prob=0.1, avg_logprob=-0.5, compression_ratio=1.5):
    return {
        "text": text,
        "no_speech_prob": no_speech_prob,
        "avg_logprob": avg_logprob,
        "compression_ratio": compression_ratio,
    }


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


def test_filter_hallucinations_convenience():
    filtered = filter_hallucinations(
        [
            make_segment("keep this"),
            make_segment("drop this", no_speech_prob=0.95),
        ]
    )
    assert len(filtered) == 1
    assert filtered[0]["text"] == "keep this"
