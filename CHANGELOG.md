# Changelog

## 0.3.3 — 2026-10-21

### Fixed
- **Silence hallucinations with a low `no_speech_prob` got through L2.**
  Over a noise bed faster-whisper filled 30 s windows with "Thank you." at
  `no_speech_prob` 0.14 and 0.75 (`avg_logprob` −0.04 / −0.22), below the
  0.8 cut and above every logprob cut, so they reached the transcript. A
  decoding window (same grouping as the 0.3.2 rescue rule) spanning ≥
  `sparse_min_duration` (10 s) with < `sparse_density` (1.0) letters/digits/CJK
  per second is now dropped whatever its metrics. Measured gap: hallucinated
  windows ≥10 s were ≤0.53 chars/s, real-speech windows ≥4.9 chars/s.
  "由 Amara.org 社群提供的字幕" (`no_speech_prob` 0.8019) no longer depends on
  that hair's-breadth margin. Without usable timing the rule does not apply.

## 0.3.2 — 2026-10-20

Behaviour fixes from the 2026-10-09 audit and 2026-10-19 probe. All change transcript output.

### Fixed
- **L4 char-loop filter rewrote numbers.** `(.{2,4})\1{2,}` ran over all
  text, so `100000000` became `100`, `0912121212` became `0912` and
  `1.000.000.000` became `1.000`. A loop whose unit is purely numeric
  (digits plus `. , : / + - $ %`, no whitespace) is now kept unless the
  repeated run is at least `char_loop_numeric_min_span` (default 20)
  characters long; a thousands group (`,000` / `.000`) is always kept;
  text loops (`xyzxyzxyz`, `哈哈哈哈哈哈`), loops that mix a digit with text
  (`第1集第1集第1集`) and runaway digit runs (`0000…` x32) are still collapsed.
  `filter_hallucinations()` now goes through the same rule.
- **L1 silence check averaged by segment count.** Three 1 s BGM tails could
  outvote 30 s of clear speech and reject the whole batch as silence
  (`filter_hallucinations()` → `[]`). The gate now uses the *smaller* of the
  plain mean and the duration-weighted mean (when every segment has a positive
  `start`/`end`), so it fires only when the batch is silent by both measures.
  Duration alone would have wiped 40 s of interview followed by 80 s of
  ambience; the min never rejects a batch the old gate passed. Without timing
  (or with non-numeric / non-finite `start`/`end`) it is the plain mean.
- **L1/L2 trusted `no_speech_prob` alone and dropped real speech.** Over music
  or a noise bed faster-whisper can report `no_speech_prob` 0.9–1.0 for a
  window of confidently decoded speech (`avg_logprob` −0.02…−0.07, 7–9
  chars/s); L2 dropped it silently (one 177 s explainer lost 20 of 173
  segments) and a short reel made only of such windows was wiped by L1. A
  high-`no_speech` decoding window — consecutive segments sharing the same
  `no_speech_prob`/`avg_logprob` — is now kept when it spans ≥
  `no_speech_rescue_min_duration` (5 s), has ≥ `no_speech_rescue_density`
  (2.0) letters/digits/CJK per second and `avg_logprob` >
  `no_speech_rescue_logprob` (−0.5); L1 counts such windows as speech.
  Whisper's own "no_speech high AND logprob < −1" was not used: measured
  silence hallucinations ("Thank you.", "由 Amara.org 社群提供的字幕") sit at
  −0.04…−0.24 and would all pass. They are 0.3–0.6 chars/s, so density keeps
  them out. Without usable timing the old rule applies unchanged.

## 0.3.1 — 2026-08-08

Packaging only — no behaviour change. Three defects that all shipped in 0.3.0.

### Fixed
- **The package reported a different version than it was published as.** The
  0.3.0 wheel on PyPI carries `Version: 0.3.0` in its metadata and
  `__version__ = "0.2.0"` inside — so `pip show whisper-guard` and
  `whisper_guard.__version__` have disagreed for two months. The number lived
  in two places (`pyproject.toml` and `__init__.py`), the repo copy was
  corrected on 2026-06-04, and the wheel had been built the day before. Nothing
  compared them.
- **The version now has one source.** hatchling reads it from
  `whisper_guard/__init__.py` (`[tool.hatch.version]`); the literal in
  `pyproject.toml` is gone, so the two cannot diverge by construction.
  `tests/test_version.py` is the belt — it compares installed metadata against
  `__version__`, which also catches a wheel built from a stale tree.
- **The sdist shipped a local virtualenv.** `whisper_guard-0.3.0.tar.gz` is
  1.5 MB, and 520 of its files are a `.venv-test/` directory — `pyvenv.cfg`
  with local absolute paths and all. `.gitignore` only listed `.venv/`, and
  hatchling does not consult `.gitignore` for sdists regardless. Both build
  targets now carry explicit excludes; the sdist is **15 KB**.

## 0.3.0 — 2026-04-15

### Added
- Dynamic logprob threshold: short segments (<1.6s) use a separate, *more lenient* `-1.7` threshold (vs `-1.5`) — genuine brief utterances naturally score lower confidence, so loosening the bar avoids wrongly dropping real short speech
- `GuardConfig.avg_logprob_short` — configurable threshold for short segments (default: `-1.7`)
- `GuardConfig.short_segment_threshold` — duration cutoff in seconds (default: `1.6`)
- 4 new test cases covering dynamic logprob behavior

### Notes
- Aligned with arkiv `transcribe.py` guard logic
- Segments without `start`/`end` timing info fall back to the normal `avg_logprob` threshold (backward compatible)

## 0.2.0 — 2026-04-08

### Fixed
- `filter_hallucinations` no longer duplicates L2 segment filtering (reuses `_filter_segments`)
- `vocab.py` default filler words aligned with spec (`嗯嗯`, `啊啊`, `呃呃`, `喔喔`)

### Changed
- Regex pattern compiled once at init instead of per-call

## 0.1.0 — 2026-04-07

### Added
- 4-layer guard: silence, segment, repetition, char-loop
- `WhisperGuard` class with `GuardConfig` / `GuardResult`
- `filter_hallucinations` convenience function
- `vocab.py`: `build_hotwords_prompt`, `filter_filler_words`
- 10 pytest cases
