
import sys
import langchain_core.messages

if "langchain.schema" not in sys.modules:
    sys.modules["langchain.schema"] = langchain_core.messages

import os
from pathlib import Path

# Disable deepeval telemetry and ensure cache dir exists
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
_cache_dir = Path(__file__).parent.parent / "data" / "deepeval"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["DEEPEVAL_CACHE_DIR"] = str(_cache_dir)

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.core.logging import get_logger
from evals.golden_dataset import GoldenRecord
from evals.pipeline_runner import PipelineResult

logger = get_logger(__name__)


@dataclass
class DeepEvalResult:
    available: bool = False
    scores: Dict[str, float] = field(default_factory=dict)
    per_sample: List[Dict] = field(default_factory=list)
    error: Optional[str] = None


def run_deepeval_eval(
    records: List[GoldenRecord],
    results: List[PipelineResult],
) -> DeepEvalResult:
    """
    Run DeepEval evaluation. Returns DeepEvalResult with available=False
    if deepeval is not installed or the eval run fails.
    """
    try:
        from deepeval import evaluate as deepeval_evaluate
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            FaithfulnessMetric,
            HallucinationMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        logger.warning("deepeval_not_installed", extra={"error": str(exc)})
        return DeepEvalResult(
            available=False, error=f"deepeval not installed: {exc}"
        )

    try:
        test_cases = []
        for record, result in zip(records, results):
            if result.error or not result.answer:
                continue
            test_cases.append(
                LLMTestCase(
                    input=record.question,
                    actual_output=result.answer,
                    expected_output=record.ground_truth,
                    retrieval_context=result.retrieved_contexts
                    if result.retrieved_contexts
                    else ["[no context retrieved]"],
                    context=record.contexts,
                )
            )

        if not test_cases:
            return DeepEvalResult(
                available=True,
                error="All pipeline runs failed — no test cases to evaluate",
            )

        metrics = [
            HallucinationMetric(threshold=0.5),
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
            ContextualPrecisionMetric(threshold=0.7),
        ]

        eval_results = deepeval_evaluate(
            test_cases=test_cases,
            metrics=metrics,
            run_async=False,
            ignore_errors=True,
        )

        per_sample = []
        metric_totals: Dict[str, List[float]] = {}

        for tc in test_cases:
            sample_scores = {}
            for metric in tc.metrics_metadata:
                name = metric.metric_name
                score = metric.score or 0.0
                sample_scores[name] = score
                metric_totals.setdefault(name, []).append(score)
            per_sample.append(
                {"question": tc.input, "scores": sample_scores}
            )

        aggregate_scores = {
            name: sum(vals) / len(vals)
            for name, vals in metric_totals.items()
            if vals
        }

        logger.info("deepeval_eval_completed", extra={"scores": aggregate_scores})
        return DeepEvalResult(
            available=True, scores=aggregate_scores, per_sample=per_sample
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("deepeval_eval_failed", extra={"error": str(exc)})
        return DeepEvalResult(available=True, error=str(exc))
