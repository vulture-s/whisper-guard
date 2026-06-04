"""Benchmark invariants as a regression guard.

The full report is `python bench/benchmark.py`. These assertions pin the
properties we never want a code change to break — chiefly that the guard does
NOT eat real speech (zero false positives). Recall is allowed to be imperfect
(fluent phantoms are a known, documented miss handled by upstream VAD), so we
only assert a floor on it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from benchmark import run_suite_a, run_suite_b  # noqa: E402


def test_no_false_positives_never_eat_real_speech():
    _, (tp, fp, tn, fn) = run_suite_a()
    assert fp == 0, f"guard dropped/altered real speech (FP={fp}) — must never happen"


def test_precision_perfect_on_corpus():
    _, (tp, fp, tn, fn) = run_suite_a()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    assert precision == 1.0, f"precision regressed to {precision:.3f}"


def test_recall_floor():
    _, (tp, fp, tn, fn) = run_suite_a()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    # fluent phantoms are a known miss; everything else must be caught
    assert recall >= 0.75, f"recall dropped below floor: {recall:.3f}"


def test_batch_rejection_perfect():
    _, correct, total = run_suite_b()
    assert correct == total, f"batch rejection regressed: {correct}/{total}"
