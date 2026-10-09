"""Class-level invariants, not single-case regressions.

Every bug fixed on 2026-10-09 (#1, #2) was the same shape: a layer that
"never eats real speech" ate real speech once the input left the shape the
bench corpus happened to have — numbers (L4), long recordings (L3), batches
that are mostly ambience (L1). The single-case regressions live next to each
fix (test_guard.py, test_repetition.py). These tests sweep the input *axis*
instead, so the next layer/threshold change that breaks the same promise for
a value nobody thought to write down still fails here.

See docs/known-bug-classes.md.
"""
import random

import pytest

from whisper_guard import GuardConfig, WhisperGuard


def _seg(text, start=None, end=None, nsp=0.1):
    seg = {"text": text, "no_speech_prob": nsp, "avg_logprob": -0.5,
           "compression_ratio": 1.5}
    if start is not None:
        seg["start"], seg["end"] = start, end
    return seg


# --- Axis 1: numbers. L4 must never rewrite a value a person said. ---------

def _numbers(seed=20261009, n=600):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        kind = rng.randrange(9)
        if kind == 0:  # plain integer, up to 19 digits
            out.append(str(rng.randrange(10 ** rng.randrange(1, 19))))
        elif kind == 1:  # adversarial: a 2-4 digit unit repeated (0912121212)
            unit = "".join(rng.choice("0123456789") for _ in range(rng.randrange(2, 5)))
            reps = rng.randrange(3, max(4, 19 // len(unit)))
            out.append(rng.choice(["", "09", "1"]) + unit * reps)
        elif kind == 2:  # IPv4, including 10.10.10.10-style repeats
            octet = str(rng.randrange(256))
            out.append(".".join(rng.choice([octet, str(rng.randrange(256))]) for _ in range(4)))
        elif kind == 3:  # thousands grouping, any length
            out.append("{:,}".format(rng.randrange(10 ** rng.randrange(4, 22))))
        elif kind == 4:  # European grouping
            out.append("{:,}".format(rng.randrange(10 ** rng.randrange(4, 16))).replace(",", "."))
        elif kind == 5:  # time code / clock
            out.append("%02d:%02d:%02d" % (rng.randrange(24), rng.randrange(60), rng.randrange(60)))
        elif kind == 6:  # date
            out.append("%04d/%02d/%02d" % (rng.randrange(1990, 2040), rng.randrange(1, 13),
                                           rng.randrange(1, 29)))
        elif kind == 7:  # version / decimal
            out.append("%d.%d.%d" % (rng.randrange(20), rng.randrange(20), rng.randrange(20)))
        else:  # money / percent
            out.append(rng.choice(["$", ""]) + str(rng.randrange(10 ** 7)) + rng.choice(["", "%"]))
    return out


def test_no_number_is_ever_rewritten_by_char_loop_removal():
    guard = WhisperGuard()
    bad = []
    for number in _numbers():
        for text in ("營收是%s元" % number, "the value is %s today" % number, number):
            cleaned, _ = guard.remove_char_loops(text)
            if number not in cleaned:
                bad.append((text, cleaned))
    assert not bad, bad[:10]


def test_numbers_survive_the_full_pipeline():
    guard = WhisperGuard()
    numbers = _numbers(n=200)
    segments = [_seg("編號 %s 已確認" % n, start=float(i * 3), end=float(i * 3 + 2.5))
                for i, n in enumerate(numbers)]
    result = guard.process(segments)
    assert result.passed, result.rejected_by
    missing = [n for n in numbers if n not in result.text]
    assert not missing, missing[:10]


def test_numeric_exemption_has_teeth():
    """The exemption must not swallow the loop class it was carved out of."""
    guard = WhisperGuard()
    for loop in ("0" * 40, "12" * 20, "第1集" * 5, "ha" * 10):
        assert guard.remove_char_loops("前 %s 後" % loop)[1] >= 1, loop


# --- Axis 2: length. L3 must judge "loop", never "long". -------------------

def _zipf(n, seed):
    rng = random.Random(seed)
    vocab = 6000
    weights = [1 / (r ** 1.07) for r in range(1, vocab + 1)]
    return rng.choices(["w%d" % i for i in range(vocab)], weights=weights, k=n)


@pytest.mark.parametrize("n_words", [30, 120, 500, 1500, 4500, 12000])
def test_natural_text_is_never_repetitive_at_any_length(n_words):
    assert WhisperGuard().is_repetitive(" ".join(_zipf(n_words, seed=n_words))) is False


@pytest.mark.parametrize("n_chars", [40, 400, 4000, 20000])
def test_unspaced_cjk_text_is_never_repetitive_at_any_length(n_chars):
    rng = random.Random(n_chars)
    pool = [chr(c) for c in range(0x4E00, 0x4E00 + 3000)]
    text = "".join(rng.choice(pool) for _ in range(n_chars))
    assert WhisperGuard().is_repetitive(text) is False


@pytest.mark.parametrize("n_words", [30, 500, 4500, 12000])
def test_a_loop_is_repetitive_at_any_length(n_words):
    words = (["thank", "you"] * n_words)[:n_words]
    assert WhisperGuard().is_repetitive(" ".join(words)) is True


# --- Axis 3: batch silence gate (L1). It drops the WHOLE batch, so it may only
# ever be *more* lenient than the plain mean, never stricter. ---------------

def test_silence_gate_never_stricter_than_plain_mean():
    guard = WhisperGuard()
    thr = guard.config.silence_threshold
    rng = random.Random(1009)
    for _ in range(2000):
        n = rng.randrange(1, 40)
        segs, t = [], 0.0
        timed = rng.random() < 0.8
        for _ in range(n):
            d = rng.choice([0.3, 1.0, 5.0, 30.0, 90.0]) * rng.random() + 0.01
            segs.append(_seg("x" if rng.random() < 0.7 else "", t if timed else None,
                             t + d if timed else None, nsp=rng.random()))
            t += d
        plain = sum(s["no_speech_prob"] for s in segs) / len(segs)
        if guard._mean_no_speech(segs) > thr:
            assert plain > thr, segs


def test_mostly_speech_by_duration_is_never_rejected_as_silence():
    """30 s of clear speech must survive any number of short silent tails."""
    guard = WhisperGuard()
    for tails in range(0, 60, 7):
        segs = [_seg("clear speech here", 0.0, 30.0, nsp=0.05)]
        segs += [_seg("", 30.0 + i, 30.5 + i, nsp=0.99) for i in range(tails)]
        result = guard.process(segs)
        assert result.rejected_by != "silence", tails
        assert "clear speech here" in result.text


def test_config_change_is_reflected_not_hardcoded():
    """Thresholds come from GuardConfig — a stricter config must bite."""
    strict = WhisperGuard(GuardConfig(silence_threshold=0.0))
    assert strict.process([_seg("hi", 0.0, 1.0, nsp=0.5)]).rejected_by == "silence"
