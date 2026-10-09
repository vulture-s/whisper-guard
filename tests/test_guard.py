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


# --- Numeric content must survive the char-loop layer (audit 2026-10-09) ---
# A repeated 2-4 char unit is a Whisper loop when it is text ("xyzxyzxyz"),
# but it is *data* when it carries digits: amounts, phone numbers, years,
# version strings. Collapsing "00" x4 in 100000000 silently rewrote the value.

import pytest


@pytest.mark.parametrize(
    "text",
    [
        "價格是 100000000 元",
        "2000000",
        "電話 0912121212",
        "1.000.000.000",
        "版本 1.1.1.1",
        "IP 10.10.10.10",
        "金額 $1,000,000,000",
        "2020202020 年",
    ],
)
def test_char_loops_do_not_rewrite_numbers(text):
    guard = WhisperGuard()
    cleaned, removed = guard.remove_char_loops(text)
    assert cleaned == text
    assert removed == 0
    assert guard.has_char_loops(text) is False


def test_numbers_survive_process_and_filter_hallucinations():
    guard = WhisperGuard()
    result = guard.process([make_segment("價格是 100000000 元")])
    assert result.text == "價格是 100000000 元"
    assert result.char_loops_removed == 0
    out = filter_hallucinations([make_segment("電話 0912121212")])
    assert [s["text"] for s in out] == ["電話 0912121212"]


def test_text_loops_still_removed_next_to_numbers():
    guard = WhisperGuard()
    cleaned, removed = guard.remove_char_loops("xyzxyzxyz 100000000")
    assert cleaned == "xyz 100000000"
    assert removed == 1
    cleaned, _ = guard.remove_char_loops("哈哈哈哈哈哈")
    assert cleaned == "哈哈"


# --- Silence layer must weight by duration, not by segment count ---


def test_silence_check_is_duration_weighted():
    # 30 s of clear speech + three 1 s BGM tails. Count-average no_speech is
    # (0.05 + 3*0.95)/4 = 0.725 > 0.6 and used to wipe the whole transcript;
    # duration-weighted it is (30*0.05 + 3*0.95)/33 = 0.13.
    segs = [make_segment("這是一段很長的正常講話內容", no_speech_prob=0.05, start=0.0, end=30.0)]
    for i in range(3):
        segs.append(make_segment("bgm", no_speech_prob=0.95, start=30.0 + i, end=31.0 + i))
    result = WhisperGuard().process(segs)
    assert result.passed is True
    assert result.rejected_by is None
    assert "這是一段很長的正常講話內容" in result.text
    assert filter_hallucinations(segs) != []


def test_silence_still_rejects_when_silent_by_count_and_duration():
    # One short clean segment + three long silent ones: silent either way.
    segs = [make_segment("hi", no_speech_prob=0.05, start=0.0, end=1.0)]
    for i in range(3):
        segs.append(make_segment("noise", no_speech_prob=0.9, start=1.0 + 10 * i, end=11.0 + 10 * i))
    result = WhisperGuard().process(segs)
    assert result.passed is False
    assert result.rejected_by == "silence"


def test_long_ambient_tail_does_not_wipe_interview():
    # 40 s of interview (8 x 5 s) then 80 s of ambience that Whisper fills with
    # three long hallucinated segments. Duration-weighted alone = 0.62 > 0.6 and
    # rejected the whole batch; the count gate (0.28) kept the interview.
    segs = [make_segment("第%d句正常回答" % i, no_speech_prob=0.05, start=5.0 * i, end=5.0 * i + 5)
            for i in range(8)]
    segs += [make_segment("Thank you for watching.", no_speech_prob=0.9, start=40.0, end=67.0),
             make_segment("Thank you.", no_speech_prob=0.9, start=67.0, end=94.0),
             make_segment("Bye.", no_speech_prob=0.9, start=94.0, end=120.0)]
    result = WhisperGuard().process(segs)
    assert result.passed is True
    assert "第0句正常回答" in result.text and "第7句正常回答" in result.text
    assert "Thank you" not in result.text  # still dropped per segment


def test_silence_without_timing_falls_back_to_plain_mean():
    segs = [
        make_segment("a", no_speech_prob=0.05),
        make_segment("b", no_speech_prob=0.95),
        make_segment("c", no_speech_prob=0.95),
        make_segment("d", no_speech_prob=0.95),
    ]
    result = WhisperGuard().process(segs)
    assert result.rejected_by == "silence"


# --- Digit-bearing hallucinations must not escape the char-loop layer ---


@pytest.mark.parametrize(
    "text,expected",
    [
        ("第1集第1集第1集第1集", "第1集"),
        ("ha1ha1ha1ha1ha1", "ha1"),
        ("0" * 32, "0000"),
        ("1.1.1.1.1.1.1.1.1.1.1.1", "1.1."),
    ],
)
def test_digit_bearing_loops_still_collapse(text, expected):
    guard = WhisperGuard()
    cleaned, removed = guard.remove_char_loops(text)
    assert removed >= 1
    assert guard.has_char_loops(text) is True
    assert cleaned.startswith(expected) and len(cleaned) < len(text)


def test_text_loops_unchanged_by_numeric_exemption():
    guard = WhisperGuard()
    for text, expected in [("哈哈哈哈哈哈哈哈", "哈哈"), ("謝謝謝謝謝謝", "謝謝"),
                           ("the the the the the", "the the"), ("ok ok ok ok ok", "ok ok")]:
        assert guard.remove_char_loops(text)[0] == expected


# --- Codex audit of #1 ---


def test_long_thousands_grouping_is_data():
    for text in ["金額 $1,000,000,000,000,000 元", "1.000.000.000.000.000.000"]:
        assert WhisperGuard().remove_char_loops(text) == (text, 0)


def test_space_separated_digit_loop_still_collapses():
    cleaned, removed = WhisperGuard().remove_char_loops("前言 0 0 0 0 0 0 結語")
    assert removed == 1 and cleaned == "前言 0 0 結語"


def test_string_timestamps_do_not_crash_and_fall_back_to_plain_mean():
    segs = [make_segment("hallucinated", no_speech_prob=0.05, start="0", end="30")]
    segs += [make_segment("noise", no_speech_prob=0.95, start=30.0 + i, end=31.0 + i) for i in range(3)]
    result = WhisperGuard().process(segs)
    assert result.rejected_by == "silence"
    assert filter_hallucinations(segs) == []
