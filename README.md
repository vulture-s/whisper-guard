# whisper-guard

`whisper-guard` is a small Python package that removes common Whisper hallucinations before you ship subtitles or transcripts downstream.

## Problem

Whisper output can degrade on silence-heavy or low-confidence clips:

- repeated phrases
- phantom subtitles on silence
- short character loops such as `哈哈哈哈` or `xyzxyzxyz`

This package extracts the anti-hallucination logic from `arkiv` into a reusable package with a minimal API.

## 4-Layer Guard

| Layer | What it does | Default |
|------|------|------|
| L1 Silence | Reject all-silence batches | avg `no_speech_prob` > `0.6` |
| L2 Segment | Filter weak segments | `no_speech_prob` > `0.8`, `avg_logprob` < `-1.5` (short <1.6s: `-1.7`), `compression_ratio` > `3.0` |
| L3 Repetition | Reject repetitive text blocks | unique chunk ratio < `0.35` |
| L4 Char loops | Remove looped patterns | 2-4 chars repeated 3+ times |

## A/B Test Results

These numbers are from the `arkiv` guard benchmark set on April 2026.

| Config | Reps | Reduction | Time |
|------|------|------|------|
| Raw Whisper | 16 | baseline | 47.7s |
| **Guard only** | **2** | **-87.5%** | **46.5s** |
| LLM polish only | 16 | 0% | 106.1s |
| Guard + LLM | 2 | -87.5% | 119.1s |
| VAD + Guard + LLM | 1 | -93.8% | 128.4s |

## Install

```bash
pip install whisper-guard
```

For local development:

```bash
pip install -e .
```

## Quick Start

```python
from faster_whisper import WhisperModel
from whisper_guard import WhisperGuard

model = WhisperModel("small")
segments, info = model.transcribe("sample.wav")

guard = WhisperGuard()
result = guard.process([segment._asdict() for segment in segments])

if result.passed:
    print(result.text)
```

## API

```python
from whisper_guard import WhisperGuard, GuardConfig, filter_hallucinations
```

`WhisperGuard.process()` expects segment dictionaries shaped like:

```python
{
    "text": "hello world",
    "no_speech_prob": 0.12,
    "avg_logprob": -0.44,
    "compression_ratio": 1.2,
    "start": 0.0,   # optional — enables dynamic logprob threshold
    "end": 2.5,      # optional — enables dynamic logprob threshold
}
```

When `start`/`end` are provided, segments shorter than `1.6s` use a stricter logprob threshold (`-1.7`) to catch hallucinations in brief audio gaps. Segments without timing info fall back to the normal threshold.

## Compatible With

- `faster-whisper`
- `openai-whisper`
- `mlx-whisper`

## Optional Vocab Helpers

```python
from whisper_guard.vocab import build_hotwords_prompt, filter_filler_words
```

## Part Of

Built for the [`arkiv`](https://github.com/vulture-s/arkiv) transcription pipeline and split out as a standalone package for reuse.

## License

MIT
