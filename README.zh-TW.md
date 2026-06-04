# whisper-guard

> [English](README.md) ｜ 繁體中文

`whisper-guard` 是一個小型 Python 套件，在你把字幕／逐字稿往下游送之前，先把 Whisper 常見的「幻覺」清掉。

## 問題

Whisper 在靜音多、信心低的片段上很容易產生幻覺輸出：

- 重複的句子（`謝謝大家 謝謝大家 謝謝大家…`）
- 靜音段冒出來的幽靈字幕（`字幕由志願者提供`、`請按讚訂閱開啟小鈴鐺`）
- 短字元迴圈（`哈哈哈哈`、`字幕由字幕由字幕由`）

這個套件把 [`arkiv`](https://github.com/vulture-s/arkiv) 裡的反幻覺邏輯抽出來，包成一個 API 極簡、可重用的套件。

## 四層防護（4-Layer Guard）

| 層級 | 做什麼 | 預設門檻 |
|------|--------|----------|
| **L1 靜音** | 整批幾乎全靜音 → 整批拒收 | 平均 `no_speech_prob` > `0.6` |
| **L2 片段** | 過濾弱片段 | `no_speech_prob` > `0.8`、`avg_logprob` < `-1.5`（短片段 <1.6s：`-1.7`）、`compression_ratio` > `3.0` |
| **L3 重複** | 整段文字過度重複 → 拒收 | 唯一片段比例 < `0.35` |
| **L4 字元迴圈** | 移除迴圈樣式 | 2-4 字元重複 3 次以上 |

> 短片段的動態門檻：當片段有提供 `start`／`end` 且長度 < 1.6s 時，`avg_logprob` 改用 `-1.7`（短音天生信心較低，避免誤殺真實短語）。沒有時間資訊的片段退回一般門檻。

## 安裝

```bash
pip install whisper-guard
```

本地開發：

```bash
pip install -e .
```

## 快速開始

```python
from faster_whisper import WhisperModel
from whisper_guard import WhisperGuard

model = WhisperModel("small")
segments, info = model.transcribe("sample.wav")

guard = WhisperGuard()
result = guard.process([segment._asdict() for segment in segments])

if result.passed:
    print(result.text)          # 清乾淨的逐字稿
else:
    print("整批被擋下：", result.rejected_by)   # silence / repetition / no_good_segments
```

## API

```python
from whisper_guard import WhisperGuard, GuardConfig, GuardResult, filter_hallucinations
```

`WhisperGuard.process()` 吃的 segment 字典長這樣：

```python
{
    "text": "今天天氣很好",
    "no_speech_prob": 0.12,
    "avg_logprob": -0.44,
    "compression_ratio": 1.2,
    "start": 0.0,   # 可選 — 有的話會啟用短片段動態門檻
    "end": 2.5,      # 可選
}
```

`GuardConfig` 的每個門檻都可以調（`silence_threshold`、`no_speech_prob`、`avg_logprob`、`repetition_threshold`…）。`filter_hallucinations(segments)` 是便利函式，直接回傳「過濾＋清乾淨」後的 segment 清單。

## 準確度 Benchmark

> ⚠️ 這是**自建標注 corpus 的 benchmark**，不是獨立第三方評測。標注反映的是「我們已知這個 guard 該抓什麼」，所以它的價值在**透明度 + 回歸守門**（程式一改、若開始吃掉真內容或放過幻覺，數字會動），不是排行榜分數。完整可重跑：`python bench/benchmark.py`。

針對一組手工標注、模擬真實中文 Whisper 輸出的 corpus（混合真實語音 + 各型態幻覺）：

| 指標 | 結果 | 意義 |
|------|------|------|
| **Precision** | **100%** | 被判幻覺的，全都真的是幻覺 |
| **假陽性率（FPR）** | **0%** | **從不誤殺真實語音**（最重要的性質） |
| **Recall** | **80%** | 抓到大多數幻覺，但**漏掉「流暢幻覺」**（見下） |
| **Batch 拒收準確度** | **100%** | L1 靜音 / L3 重複的整批判斷全對 |

## 已知極限

whisper-guard 是 **metric / pattern 驅動**的——它看的是 `no_speech_prob`、`avg_logprob`、`compression_ratio` 和字元迴圈樣式。所以它**抓不到「指標健康的流暢幻覺」**：一句語法正確、單次出現、信心分數正常的幻覺句（例如靜音段冒出的「請按讚訂閱開啟小鈴鐺」），用指標看跟真實語音無法區分。

→ 這類要靠**上游 VAD**（語音活動偵測，先把靜音段切掉）來擋。whisper-guard 設計上就是跟 VAD 搭配：VAD 擋靜音幻覺、whisper-guard 擋重複／低信心／迴圈型幻覺。Benchmark 的 80% recall 就是誠實地把這個漏接反映出來。

## 相容於

- `faster-whisper`
- `openai-whisper`
- `mlx-whisper`

## 詞彙輔助（選用）

```python
from whisper_guard.vocab import build_hotwords_prompt, filter_filler_words

build_hotwords_prompt(["孟竹", "Furutech", "arkiv"])   # 組 Whisper hotwords prompt
filter_filler_words("嗯嗯這個啊啊那個")                  # 去贅詞 → "這個 那個"
```

## 來源

為 [`arkiv`](https://github.com/vulture-s/arkiv) 轉錄管線打造，抽成獨立套件供重用。

## 授權

MIT
