# Changelog

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
