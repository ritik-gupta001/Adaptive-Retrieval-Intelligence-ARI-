"""
Unit tests for Module 13.
RAGAS and DeepEval wrappers are NOT tested here (they require LLM API
keys and a real vector store — they belong in CI with secrets, not in
the fast local unit suite). We test:
  - golden dataset loading and validation (pure I/O with temp files)
  - _compute_our_metrics (pure function, all arithmetic)
  - build_report (pure function, composition)
  - EvalReport.summary_lines() (output formatting)
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.append(".")

from evals.golden_dataset import GoldenRecord, load_golden_dataset
from evals.pipeline_runner import PipelineResult
from evals.report import _compute_our_metrics, build_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_records(n=3):
    return [
        GoldenRecord(
            question=f"question {i}",
            ground_truth=f"ground truth {i}",
            contexts=[f"context {i}"],
            expected_strategy="vector_search",
            expected_intent="factual",
        )
        for i in range(n)
    ]


def _make_result(
    question="question 0",
    strategies=None,
    confidence=0.85,
    confidence_level="high",
    retry_count=0,
    hallucinations=None,
    clarification=False,
    error=None,
    num_docs=3,
):
    return PipelineResult(
        question=question,
        answer=f"answer to {question}",
        citations=["doc1"],
        retrieved_contexts=[f"ctx{i}" for i in range(num_docs)],
        strategies_used=strategies or ["vector_search"],
        confidence_score=confidence,
        confidence_level=confidence_level,
        hallucination_risk=0.0,
        retry_count=retry_count,
        reflection_is_supported=True,
        reflection_hallucinations=hallucinations or [],
        clarification_needed=clarification,
        error=error,
    )


# ---------------------------------------------------------------------------
# golden_dataset.py
# ---------------------------------------------------------------------------

def test_load_golden_dataset_reads_valid_jsonl():
    records = [
        {
            "question": "What is X?",
            "ground_truth": "X is a thing.",
            "contexts": ["X context"],
            "expected_strategy": "vector_search",
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        path = Path(f.name)

    loaded = load_golden_dataset(path)
    assert len(loaded) == 1
    assert loaded[0].question == "What is X?"
    assert loaded[0].expected_strategy == "vector_search"


def test_load_golden_dataset_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_golden_dataset(Path("/nonexistent/golden.jsonl"))


def test_load_golden_dataset_raises_on_empty_question():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({
            "question": "",   # empty — should fail
            "ground_truth": "truth",
            "contexts": ["ctx"],
        }) + "\n")
        path = Path(f.name)

    with pytest.raises(ValueError, match="line 1"):
        load_golden_dataset(path)


def test_load_golden_dataset_raises_on_empty_contexts():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({
            "question": "q",
            "ground_truth": "truth",
            "contexts": [],   # empty list — should fail
        }) + "\n")
        path = Path(f.name)

    with pytest.raises(ValueError):
        load_golden_dataset(path)


def test_load_golden_dataset_raises_on_invalid_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("not valid json\n")
        path = Path(f.name)

    with pytest.raises(ValueError, match="line 1"):
        load_golden_dataset(path)


def test_load_golden_dataset_skips_blank_lines():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n")
        f.write(json.dumps({"question": "q", "ground_truth": "g", "contexts": ["c"]}) + "\n")
        f.write("\n")
        path = Path(f.name)

    loaded = load_golden_dataset(path)
    assert len(loaded) == 1


# ---------------------------------------------------------------------------
# _compute_our_metrics
# ---------------------------------------------------------------------------

def test_our_metrics_perfect_router_accuracy():
    records = _make_records(3)
    results = [_make_result(strategies=["vector_search"]) for _ in range(3)]
    metrics = _compute_our_metrics(records, results)
    assert metrics.router_accuracy == pytest.approx(1.0)


def test_our_metrics_zero_router_accuracy():
    records = _make_records(3)
    results = [_make_result(strategies=["web_search"]) for _ in range(3)]
    metrics = _compute_our_metrics(records, results)
    assert metrics.router_accuracy == pytest.approx(0.0)


def test_our_metrics_partial_router_accuracy():
    records = _make_records(4)
    results = [
        _make_result(strategies=["vector_search"]),  # correct
        _make_result(strategies=["vector_search"]),  # correct
        _make_result(strategies=["web_search"]),     # wrong
        _make_result(strategies=["web_search"]),     # wrong
    ]
    metrics = _compute_our_metrics(records, results)
    assert metrics.router_accuracy == pytest.approx(0.5)


def test_our_metrics_hallucination_rate():
    records = _make_records(4)
    results = [
        _make_result(hallucinations=["fake claim"]),  # hallucinated
        _make_result(hallucinations=[]),               # clean
        _make_result(hallucinations=["another fake"]), # hallucinated
        _make_result(hallucinations=[]),               # clean
    ]
    metrics = _compute_our_metrics(records, results)
    assert metrics.hallucination_rate == pytest.approx(0.5)


def test_our_metrics_failed_runs_counted_separately():
    records = _make_records(3)
    results = [
        _make_result(),
        _make_result(error="provider outage"),
        _make_result(),
    ]
    metrics = _compute_our_metrics(records, results)
    assert metrics.failed_runs == 1
    assert metrics.total_runs == 3


def test_our_metrics_empty_results():
    metrics = _compute_our_metrics([], [])
    assert metrics.total_runs == 0
    assert metrics.router_accuracy == 0.0


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

def test_build_report_assembles_correctly():
    records = _make_records(2)
    results = [_make_result() for _ in range(2)]
    report = build_report(records, results)

    assert report.total_queries == 2
    assert report.our_metrics.total_runs == 2
    assert len(report.per_sample) == 2
    assert report.ragas_scores == {}
    assert report.deepeval_scores == {}


def test_build_report_per_sample_strategy_correct_flag():
    records = _make_records(2)
    results = [
        _make_result(strategies=["vector_search"]),  # matches expected
        _make_result(strategies=["web_search"]),     # does NOT match
    ]
    report = build_report(records, results)

    assert report.per_sample[0]["strategy_correct"] is True
    assert report.per_sample[1]["strategy_correct"] is False


def test_build_report_includes_ragas_scores_when_available():
    from evals.ragas_eval import RagasEvalResult

    records = _make_records(1)
    results = [_make_result()]
    ragas_result = RagasEvalResult(
        available=True,
        scores={"faithfulness": 0.92, "answer_relevancy": 0.88},
    )
    report = build_report(records, results, ragas_result=ragas_result)
    assert report.ragas_scores["faithfulness"] == 0.92


def test_build_report_excludes_ragas_scores_on_error():
    from evals.ragas_eval import RagasEvalResult

    records = _make_records(1)
    results = [_make_result()]
    ragas_result = RagasEvalResult(
        available=True, scores={}, error="OpenAI rate limit"
    )
    report = build_report(records, results, ragas_result=ragas_result)
    assert report.ragas_scores == {}
    assert report.ragas_error == "OpenAI rate limit"


# ---------------------------------------------------------------------------
# EvalReport.summary_lines
# ---------------------------------------------------------------------------

def test_summary_lines_contains_key_fields():
    records = _make_records(3)
    results = [_make_result() for _ in range(3)]
    report = build_report(records, results)
    summary = "\n".join(report.summary_lines())

    assert "Router Accuracy" in summary
    assert "Avg Confidence Score" in summary
    assert "Hallucination Rate" in summary
    assert "ARI EVALUATION REPORT" in summary


def test_summary_lines_shows_ragas_skipped_message_when_not_run():
    records = _make_records(1)
    results = [_make_result()]
    report = build_report(records, results)  # no ragas_result passed
    summary = "\n".join(report.summary_lines())
    assert "RAGAS" in summary
