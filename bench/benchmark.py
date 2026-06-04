#!/usr/bin/env python3
"""whisper-guard accuracy benchmark.

This is a **curated fixture benchmark**, not an independent third-party one. The
corpus below is hand-labelled with realistic Whisper-on-Chinese hallucination
patterns (phantom subtitle credits, low-confidence garble, repetition, character
loops) alongside genuine speech. It measures the guard as a binary classifier and
doubles as a regression guard — if a code change quietly starts eating real
speech (false positives) or letting hallucinations through, the numbers move.

It does NOT prove real-world accuracy on arbitrary audio; the labels reflect what
we already expect the guard to catch. Read it as transparency + regression, not
as a leaderboard score.

Run:  python bench/benchmark.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from whisper_guard import WhisperGuard, filter_hallucinations


# --------------------------------------------------------------------------
# Suite A — per-segment classifier (L2 segment filter + L4 char-loop cleaning)
#
# A realistic transcript is mostly good speech with a few hallucinations mixed
# in, so we keep avg no_speech_prob low enough that L1 (silence batch reject)
# does not fire — letting us measure per-segment behaviour. A segment is
# "predicted hallucination" if the guard drops it OR rewrites its text.
# --------------------------------------------------------------------------
@dataclass
class Seg:
    text: str
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float
    is_hallucination: bool          # ground-truth label
    kind: str                       # for the per-kind breakdown
    start: Optional[float] = None
    end: Optional[float] = None

    def as_dict(self) -> dict:
        d = {
            "text": self.text,
            "no_speech_prob": self.no_speech_prob,
            "avg_logprob": self.avg_logprob,
            "compression_ratio": self.compression_ratio,
        }
        if self.start is not None and self.end is not None:
            d["start"], d["end"] = self.start, self.end
        return d


def good(text, nsp=0.15, lp=-0.5, cr=1.4, kind="clean_speech", start=None, end=None):
    return Seg(text, nsp, lp, cr, False, kind, start, end)


def bad(text, nsp=0.15, lp=-0.5, cr=1.4, kind="", start=None, end=None):
    return Seg(text, nsp, lp, cr, True, kind, start, end)


SEGMENTS: List[Seg] = [
    # --- genuine speech (must be kept, unchanged) ---
    good("今天天氣很好，我們一起去公園散步聊天。"),
    good("這支麥克風的低頻量感比上一代飽滿不少。"),
    good("好，那我們先從第一個議題開始討論。"),
    good("我覺得這個方案在成本上還是有優勢的。"),
    good("謝謝你的提問，這部分我等一下會說明。"),
    good("等一下記得把檔案上傳到共用資料夾。"),
    good("這次拍攝的色溫大概落在五千六百K左右。"),
    good("對，就是這個位置，再往左邊一點點。"),
    good("欸這個之前有講過，你還記得嗎？"),
    good("好的沒問題，我下午就把報告寄給你。"),
    # genuine short segments (low logprob is normal for brief audio) — must keep
    good("對啊。", lp=-1.6, kind="clean_short", start=4.0, end=4.7),
    good("好。", lp=-1.65, kind="clean_short", start=8.0, end=8.5),
    good("嗯，是。", lp=-1.55, kind="clean_short", start=12.0, end=12.6),

    # --- L2: phantom subtitle credits on silence (high no_speech_prob) ---
    bad("字幕由志願者提供", nsp=0.92, kind="phantom_credit"),
    bad("請不吝點贊訂閱轉發打賞支持明鏡與點點欄目", nsp=0.88, kind="phantom_credit"),
    bad("字幕由 Amara.org 社群提供", nsp=0.95, kind="phantom_credit"),
    bad("本字幕由公眾號搬運", nsp=0.86, kind="phantom_credit"),

    # --- L2: low-confidence garble (avg_logprob well below -1.5) ---
    bad("呃那個就是那個", lp=-2.1, kind="low_logprob"),
    bad("欸欸欸這個", lp=-2.4, kind="low_logprob"),
    bad("唔嗯啊", lp=-1.95, kind="low_logprob"),

    # --- L2: high compression ratio (internally repetitive) ---
    bad("的的的的的的的的", cr=4.5, lp=-0.9, kind="high_compression"),
    bad("好的好的好的好的好的好的", cr=3.8, lp=-0.8, kind="high_compression"),

    # --- L4: character loops (good metrics, must be CLEANED not dropped) ---
    bad("哈哈哈哈哈哈哈哈", kind="char_loop"),
    bad("字幕由字幕由字幕由", kind="char_loop"),
    bad("然後然後然後然後", kind="char_loop"),

    # --- KNOWN LIMITATION: fluent phantoms with healthy metrics ---
    # A single, grammatical hallucinated sentence on near-silence looks identical
    # to real speech by metrics alone (low no_speech_prob, fine logprob, no loop).
    # whisper-guard is metric/pattern-based and CANNOT catch these — that's what
    # upstream VAD is for. We label them hallucination on purpose so the miss
    # shows up honestly in recall. Expected: guard keeps them (false negatives).
    bad("請按讚訂閱開啟小鈴鐺。", nsp=0.28, lp=-0.5, cr=1.5, kind="fluent_phantom"),
    bad("以上就是今天的內容，感謝收看。", nsp=0.3, lp=-0.5, cr=1.6, kind="fluent_phantom"),
    bad("謝謝大家，我們下次再見。", nsp=0.25, lp=-0.45, cr=1.5, kind="fluent_phantom"),
]


def run_suite_a():
    in_segs = [s.as_dict() for s in SEGMENTS]
    avg_nsp = sum(s["no_speech_prob"] for s in in_segs) / len(in_segs)
    assert avg_nsp <= 0.6, f"corpus would trip L1 silence reject (avg nsp={avg_nsp:.2f})"

    kept = filter_hallucinations(in_segs)
    # match kept back to inputs by identity of original text
    kept_by_text = {}
    for k in kept:
        kept_by_text.setdefault(k["text"], []).append(k)

    rows = []
    tp = fp = tn = fn = 0
    for seg in SEGMENTS:
        survivors = kept_by_text.get(seg.text)
        if survivors is None:
            # not in output at all → dropped
            predicted = True
            action = "dropped"
        else:
            survivor = survivors[0]
            if survivor["text"] != seg.text:
                predicted = True
                action = f"cleaned → {survivor['text']!r}"
            else:
                predicted = False
                action = "kept"
        # also treat dropped-and-also-cleaned cases as predicted hallucination (already covered)
        label = seg.is_hallucination
        if label and predicted:
            tp += 1
        elif (not label) and predicted:
            fp += 1
        elif (not label) and (not predicted):
            tn += 1
        else:
            fn += 1
        rows.append((seg.kind, label, predicted, action, seg.text))
    return rows, (tp, fp, tn, fn)


# --------------------------------------------------------------------------
# Suite B — batch-level rejection (L1 silence + L3 repetition)
# Each batch is labelled should_reject; we check WhisperGuard().process().passed
# --------------------------------------------------------------------------
def seglist(texts, nsp=0.15, lp=-0.5, cr=1.4):
    return [{"text": t, "no_speech_prob": nsp, "avg_logprob": lp, "compression_ratio": cr} for t in texts]


BATCHES = [
    ("silence_batch", True, seglist(["呃", "嗯", "啊"], nsp=0.9)),
    ("repetition_batch", True, seglist(["謝謝大家"] * 8)),
    ("subtitle_credit_loop", True, seglist(["字幕由志願者提供"] * 6, nsp=0.7)),
    ("normal_conversation", False, seglist([
        "今天我們來聊聊這支耳機的調音取向。",
        "它的中頻人聲位置比較靠前，適合聽流行。",
        "低頻收得算快，不會糊成一團。",
    ])),
    ("normal_with_one_phantom", False, seglist([
        "好我們開始錄囉。",
        "這段先講重點，細節等下補。",
    ]) + seglist(["字幕由志願者提供"], nsp=0.92)),
    ("short_meeting_note", False, seglist([
        "下週三開會。",
        "記得帶筆電。",
    ])),
]


def run_suite_b():
    guard = WhisperGuard()
    rows = []
    correct = 0
    for name, should_reject, batch in BATCHES:
        result = guard.process(batch)
        rejected = not result.passed
        ok = rejected == should_reject
        correct += ok
        rows.append((name, should_reject, rejected, result.rejected_by, ok))
    return rows, correct, len(BATCHES)


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def main():
    print("=" * 72)
    print("whisper-guard accuracy benchmark  (curated fixture corpus)")
    print("=" * 72)

    # Suite A
    rows, (tp, fp, tn, fn) = run_suite_a()
    print("\n## Suite A — per-segment classifier (L2 filter + L4 char-loop)\n")
    print(f"{'kind':<18}{'label':<8}{'pred':<8}{'action':<22}text")
    print("-" * 72)
    for kind, label, pred, action, text in rows:
        lt = "halluc" if label else "good"
        pt = "halluc" if pred else "good"
        flag = "" if label == pred else "  <-- MISS"
        disp = (text[:14] + "…") if len(text) > 15 else text
        print(f"{kind:<18}{lt:<8}{pt:<8}{action[:20]:<22}{disp}{flag}")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tp + fp + tn + fn
    print("-" * 72)
    print(f"segments: {total}   TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"precision={pct(tp, tp + fp)}   recall={pct(tp, tp + fn)}   "
          f"false-positive-rate={pct(fp, fp + tn)}   F1={f1:.3f}")

    # Suite B
    brows, bcorrect, btotal = run_suite_b()
    print("\n## Suite B — batch rejection (L1 silence + L3 repetition)\n")
    print(f"{'batch':<26}{'should_reject':<15}{'rejected':<10}{'reason':<18}ok")
    print("-" * 72)
    for name, should, rejected, reason, ok in brows:
        print(f"{name:<26}{str(should):<15}{str(rejected):<10}{str(reason or '-'):<18}{'✓' if ok else '✗'}")
    print("-" * 72)
    print(f"batch accuracy: {bcorrect}/{btotal} = {pct(bcorrect, btotal)}")

    print("\n" + "=" * 72)
    print(f"SUMMARY  Suite A: precision {pct(tp, tp + fp)} / recall {pct(tp, tp + fn)} "
          f"/ FPR {pct(fp, fp + tn)}   |   Suite B: {pct(bcorrect, btotal)} batch accuracy")
    print("=" * 72)


if __name__ == "__main__":
    main()
