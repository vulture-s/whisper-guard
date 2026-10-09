from whisper_guard import WhisperGuard


def make_segment(text, start, end):
    return {
        "text": text,
        "no_speech_prob": 0.1,
        "avg_logprob": -0.5,
        "compression_ratio": 1.5,
        "start": start,
        "end": end,
    }


# --- L3 must not reject long genuine transcripts (audit 2026-10-09) ---
# Type/token ratio falls with length for ANY natural text (Zipf's law): ~0.64
# at 100 words, ~0.36 at 1500, ~0.27 at 4500. A whole-transcript ratio below
# 0.35 therefore said "this is long", not "this is a loop": a ~10-minute
# English recording came back as text="" / rejected_by="repetition".


def _zipf_words(n, seed=7, vocab=6000):
    import random

    rng = random.Random(seed)
    weights = [1 / (rank ** 1.07) for rank in range(1, vocab + 1)]
    return rng.choices(["w%d" % i for i in range(vocab)], weights=weights, k=n)


def _as_segments(words, per_segment=15):
    return [
        make_segment(" ".join(words[i:i + per_segment]), start=float(i), end=float(i) + 5.0)
        for i in range(0, len(words), per_segment)
    ]


def test_long_natural_transcript_not_rejected_as_repetition():
    words = _zipf_words(4500)
    assert len(set(words)) / len(words) < 0.35  # the old whole-text test fired here
    guard = WhisperGuard()
    assert guard.is_repetitive(" ".join(words)) is False
    result = guard.process(_as_segments(words))
    assert result.passed is True
    assert result.rejected_by is None
    # (L4 may still collapse a few adjacent identical draws like "w0 w0 w0")
    assert len(result.text.split()) > 4000


def test_long_transcript_that_is_mostly_a_loop_still_rejected():
    words = _zipf_words(300) + ["thank", "you"] * 1500
    guard = WhisperGuard()
    assert guard.is_repetitive(" ".join(words)) is True


def test_short_text_word_ratio_behaviour_unchanged():
    guard = WhisperGuard()
    assert guard.is_repetitive("abc123 " * 12) is True
    assert guard.is_repetitive("the quick brown fox jumps over the lazy dog") is False
