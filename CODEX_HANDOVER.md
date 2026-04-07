# Handover: whisper-guard 獨立 Package 建置

**Generated**: 2026-04-07 12:45
**From**: Claude Code CLI
**To**: Codex
**Project root**: `C:\Users\user\Desktop\whisper-guard\`

---

## Goal

從 arkiv 的 `transcribe.py` 抽出四層防幻覺 Guard 邏輯，建立獨立 Python package `whisper-guard`，含完整 tests + README + pyproject.toml，準備推上 GitHub `vulture-s/whisper-guard`。

---

## Status

**Last completed step**: Plan 完成 + A/B test 數據齊全。尚未寫任何程式碼。

---

## Completed

- [x] A/B test 跑完（PC RTX 4070 + Mac M2 Max 跨平台驗證）
- [x] Guard 邏輯確認（4 層，87-93% 幻覺消除率）
- [x] API 設計確認（`WhisperGuard`, `GuardConfig`, `GuardResult`, `filter_hallucinations`）
- [x] Roadmap Phase 10 規劃確認
- [x] Repo 結構設計確認
- [x] 建立 `C:\Users\user\Desktop\whisper-guard\` 目錄

---

## Pending

1. **建立 `whisper_guard/guard.py`** — 四層 Guard 核心邏輯（從下方原始碼重構）
2. **建立 `whisper_guard/vocab.py`** — hotwords + filter dictionary（optional module）
3. **建立 `whisper_guard/__init__.py`** — 匯出 public API
4. **建立 `tests/test_guard.py`** — ≥ 6 個 test cases
5. **建立 `pyproject.toml`** — package metadata
6. **建立 `README.md`** — 英文，含 A/B test 數據
7. **建立 `LICENSE`** — MIT
8. **建立 `.gitignore`** — 標準 Python

---

## Blocked / Notes

- **不要發 PyPI** — 先建 repo，手動推 GitHub
- **不要建 GitHub repo** — 我會手動建 `vulture-s/whisper-guard`
- **不要加 CI/CD、pre-commit、mypy** — v0.1.0 保持最小化
- **零外部依賴** — 只用 `re`（stdlib）
- **Python ≥ 3.9**

---

## Key Files

| File | Purpose |
|------|---------|
| `C:\Users\user\.arkiv\transcribe.py` L172-231 | Guard 原始碼（下方已複製） |
| `C:\Users\user\.arkiv\config.py` | 閾值參考 |
| `C:\Users\user\.arkiv\bench_guard_ab_results.json` | A/B test 數據（README 用） |

---

## Environment Context

```
# 不需要 SSH 或服務
# 純本地 Python package 建置
Python >= 3.9
測試框架: pytest
```

---

## Suggested First Commands

```bash
cd C:\Users\user\Desktop\whisper-guard
# 實作完成後：
pip install -e .
python -c "from whisper_guard import WhisperGuard; print('OK')"
pytest tests/ -v
```

---

## Platform-Specific Notes

- **Target is Codex**: 所有檔案在 `C:\Users\user\Desktop\whisper-guard\` 下建立
- 下方包含完整的原始碼、API 設計、測試規格、README 內容規格，直接實作即可

---

## Implementation Spec

### A. 原始 Guard 邏輯（從 arkiv transcribe.py 抽出）

```python
# transcribe.py lines 172-231 — 這是要重構的原始碼

NO_SPEECH_THRESHOLD = 0.6

def _postprocess(text: str, lang: str, segments: list, language: str) -> tuple:
    if not segments:
        return text, lang

    # Guard 1: ALL segments are silence
    avg_no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
    if avg_no_speech > NO_SPEECH_THRESHOLD:
        return "", lang

    # Guard 2: Per-segment filtering
    good_segments = []
    for s in segments:
        seg_text = s.get("text", "").strip()
        if not seg_text:
            continue
        if s.get("no_speech_prob", 0) > 0.8:
            continue
        if s.get("avg_logprob", 0) < -1.5:
            continue
        if s.get("compression_ratio", 1) > 3.0:
            continue
        good_segments.append(seg_text)

    if not good_segments:
        return "", lang

    filtered_text = " ".join(good_segments).strip()

    # Guard 3: Text-level repetition
    if _is_repetitive(filtered_text):
        return "", lang

    # Guard 4: Character-level loops
    if _has_char_loops(filtered_text):
        filtered_text = _remove_char_loops(filtered_text)

    return filtered_text, lang


def _is_repetitive(text: str, window: int = 6, threshold: float = 0.35) -> bool:
    if len(text) < window * 3:
        return False
    chunks = [text[i:i+window] for i in range(0, len(text) - window, window)]
    unique = len(set(chunks))
    return unique / len(chunks) < threshold


def _has_char_loops(text: str, min_pattern: int = 2, min_repeats: int = 3) -> bool:
    import re
    return bool(re.search(r'(.{2,4})\1{2,}', text))


def _remove_char_loops(text: str) -> str:
    import re
    return re.sub(r'(.{2,4})\1{2,}', r'\1', text)
```

### B. 重構後的 API 設計

#### `whisper_guard/__init__.py`

```python
from .guard import WhisperGuard, GuardConfig, GuardResult, filter_hallucinations

__version__ = "0.1.0"
__all__ = ["WhisperGuard", "GuardConfig", "GuardResult", "filter_hallucinations"]
```

#### `whisper_guard/guard.py`

用 `dataclass` 實作。Python 3.9 相容 — 用 `from typing import Optional, List`，不用 `str | None` 或 `list[dict]`。核心 class：

```python
@dataclass
class GuardConfig:
    silence_threshold: float = 0.6       # L1
    no_speech_prob: float = 0.8          # L2
    avg_logprob: float = -1.5            # L2
    compression_ratio: float = 3.0       # L2
    repetition_window: int = 6           # L3
    repetition_threshold: float = 0.35   # L3
    char_loop_min_pattern: int = 2       # L4
    char_loop_min_repeats: int = 3       # L4

@dataclass
class GuardResult:
    text: str
    passed: bool
    rejected_by: Optional[str] = None     # "silence" | "no_good_segments" | "repetition"
    original_count: int = 0
    filtered_count: int = 0
    char_loops_removed: int = 0

class WhisperGuard:
    def __init__(self, config: "Optional[GuardConfig]" = None): ...
    def process(self, segments: "List[dict]") -> GuardResult: ...
    def is_repetitive(self, text: str) -> bool: ...
    def has_char_loops(self, text: str) -> bool: ...
    def remove_char_loops(self, text: str) -> str: ...

def filter_hallucinations(segments: "List[dict]", config: "Optional[GuardConfig]" = None) -> "List[dict]":
    """Convenience function — returns only clean segments."""
```

Segment dict 格式（輸入規格）：
```python
{"text": str, "no_speech_prob": float, "avg_logprob": float, "compression_ratio": float}
```

#### `whisper_guard/vocab.py`

```python
def build_hotwords_prompt(words: list[str], language: str = "zh") -> str:
    """Build initial_prompt for Whisper with hotwords."""
    return " ".join(words)

def filter_filler_words(text: str, dictionary: list[str] | None = None) -> str:
    """Remove filler words. Default Chinese: 嗯嗯, 啊啊, 呃呃, 喔喔"""
```

### C. Tests 規格（`tests/test_guard.py`）

用 pytest，至少 10 個 test cases：

1. `test_normal_segments_pass` — 正常 segments 全部通過
2. `test_silence_rejection` — 全靜音 → L1 rejected_by="silence"
3. `test_high_no_speech_filtered` — 單個高 no_speech segment 被 L2 濾掉
4. `test_low_logprob_filtered` — 低 logprob segment 被 L2 濾掉
5. `test_compression_ratio_filtered` — 高 compression segment 被 L2 濾掉
6. `test_repetitive_text_rejected` — 高重複文字 → L3 rejected_by="repetition"
7. `test_char_loops_removed` — 字元迴圈被 L4 偵測並移除
8. `test_empty_segments` — 空 list → passed=False, text=""
9. `test_custom_config` — 自訂閾值被尊重
10. `test_filter_hallucinations_convenience` — convenience function 回傳乾淨 segment list

Helper function：
```python
def make_segment(text, no_speech_prob=0.1, avg_logprob=-0.5, compression_ratio=1.5):
    return {"text": text, "no_speech_prob": no_speech_prob,
            "avg_logprob": avg_logprob, "compression_ratio": compression_ratio}
```

### D. pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "whisper-guard"
version = "0.1.0"
description = "4-layer anti-hallucination filter for Whisper speech-to-text output"
readme = "README.md"
license = "MIT"
requires-python = ">=3.9"
authors = [{ name = "Hevin Yeh", email = "vulture.s.tw@gmail.com" }]
keywords = ["whisper", "speech-to-text", "hallucination", "filter", "asr"]
classifiers = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
]

[project.urls]
Homepage = "https://github.com/vulture-s/whisper-guard"
Repository = "https://github.com/vulture-s/whisper-guard"
```

### E. README.md 內容規格（英文）

必須包含：

1. **Title**: `# whisper-guard` + 一句話描述
2. **Problem**: Whisper hallucinations（repeated phrases, phantom subtitles, char loops）
3. **4-Layer Guard 表格**:

| Layer | What it does | Default |
|-------|-------------|---------|
| L1 Silence | Reject all-silence | avg no_speech > 0.6 |
| L2 Segment | Filter bad segments | no_speech > 0.8, logprob < -1.5, compression > 3.0 |
| L3 Repetition | Reject repetitive text | < 35% unique chunks |
| L4 Char loops | Remove pattern loops | pattern repeated 3+ times |

4. **A/B Test Results**（真實數據）:

| Config | Reps | Reduction | Time |
|--------|------|-----------|------|
| Raw Whisper | 16 | — | 47.7s |
| **Guard only** | **2** | **-87.5%** | **46.5s** |
| LLM polish only | 16 | 0% | 106.1s |
| Guard + LLM | 2 | -87.5% | 119.1s |
| VAD + Guard + LLM | 1 | -93.8% | 128.4s |

5. **Install**: `pip install whisper-guard`
6. **Quick Start**: 5-line example with faster-whisper
7. **Compatible with**: faster-whisper, openai-whisper, mlx-whisper
8. **Part of**: Link to arkiv repo

### F. LICENSE — MIT, Copyright (c) 2026 Hevin Yeh

### G. .gitignore — 標準 Python（`__pycache__/`, `*.pyc`, `*.egg-info/`, `dist/`, `build/`, `.venv/`, `.pytest_cache/`）

---

## Session Log

### 2026-04-07 12:45 — Claude Code CLI
- 完成 A/B test（PC + Mac 跨平台）
- 完成 plan + handover 文件
- 交接給 Codex 執行實作
