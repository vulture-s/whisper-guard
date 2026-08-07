# Changelog

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
