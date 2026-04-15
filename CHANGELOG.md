# Changelog

## 0.3.0 — 2026-04-15

### Added
- Dynamic logprob threshold: short segments (<1.6s) use stricter `-1.7` threshold to catch hallucinations that tend to appear in brief audio gaps
- `GuardConfig.avg_logprob_short` — configurable stricter threshold for short segments (default: `-1.7`)
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
