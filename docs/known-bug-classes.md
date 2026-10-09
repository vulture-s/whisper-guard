# Known bug classes

Every guard layer promises "never eat real speech". Each bug so far broke that
promise for an input *shape* the bench corpus did not contain. When you touch a
layer or a threshold, the matching row below must stay green — and if you find
a new shape, add an axis to `tests/test_invariants.py`, not just one case.

| Class | Layer | Fixed in | Single-case regression | Class-level test (`tests/test_invariants.py`) |
|---|---|---|---|---|
| Numbers read as character loops (`0912121212` → `0912`, `1000000` → `100`) | L4 | #1 | `test_guard.py::test_char_loops_do_not_rewrite_numbers` and neighbours | `test_no_number_is_ever_rewritten_by_char_loop_removal`, `test_numbers_survive_the_full_pipeline`, `test_numeric_exemption_has_teeth` |
| A whole-text statistic that drifts with length (type/token ratio < 0.35 at ~1500 words → whole transcript rejected) | L3 | #2 | `test_repetition.py::test_long_natural_transcript_not_rejected_as_repetition` | `test_natural_text_is_never_repetitive_at_any_length`, `test_unspaced_cjk_text_is_never_repetitive_at_any_length`, `test_a_loop_is_repetitive_at_any_length` |
| A batch-level gate outvoted by many short silent segments | L1 | #1 | `test_guard.py::test_silence_check_is_duration_weighted`, `test_long_ambient_tail_does_not_wipe_interview` | `test_silence_gate_never_stricter_than_plain_mean`, `test_mostly_speech_by_duration_is_never_rejected_as_silence` |

Rules of thumb that came out of these:

- A gate that drops the **whole batch** must be at least as lenient as the
  per-segment filters behind it; prefer letting L2 drop segments one by one.
- Any ratio computed over the whole transcript must be checked at 10× the
  length you tested it on. Use windows.
- An exemption needs a test that it still catches the thing it was carved out
  of (`test_numeric_exemption_has_teeth`).

Known gap: L1 without timestamps still uses the plain segment-count mean, so a
batch of untimed empty segments can still outvote a few real lines.

There is no CI in this repo; `python -m pytest` from the repo root runs all of
the above.
