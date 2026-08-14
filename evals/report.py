
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.core.logging import get_logger
from evals.golden_dataset import GoldenRecord
from evals.pipeline_runner import PipelineResult

logger = get_logger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"


@dataclass
class OurMetrics:
    """System-specific metrics not captured by RAGAS or DeepEval."""
    router_accuracy: float          # fraction that used the expected_strategy
    avg_confidence_score: float     # mean confidence across runs
    avg_retry_count: float          # mean retries (ideally close to 0)
    hallucination_rate: float       # fraction with hallucinations detected
    clarification_rate: float       # fraction that triggered clarification
    avg_docs_retrieved: int         # mean number of docs retrieved
    failed_runs: int                # pipeline runs that raised exceptions
    total_runs: int


@dataclass
class EvalReport:
    timestamp: str
    total_queries: int
    our_metrics: OurMetrics
    ragas_scores: Dict[str, float] = field(default_factory=dict)
    ragas_error: Optional[str] = None
    deepeval_scores: Dict[str, float] = field(default_factory=dict)
    deepeval_error: Optional[str] = None
    per_sample: List[Dict] = field(default_factory=list)

    def summary_lines(self) -> List[str]:
        lines = [
            "=" * 60,
            "ARI EVALUATION REPORT",
            f"Timestamp : {self.timestamp}",
            f"Queries   : {self.total_queries}",
            "=" * 60,
            "",
            "── Our Metrics ──────────────────────────────────────────",
            f"  Router Accuracy      : {self.our_metrics.router_accuracy:.1%}",
            f"  Avg Confidence Score : {self.our_metrics.avg_confidence_score:.3f}",
            f"  Avg Retry Count      : {self.our_metrics.avg_retry_count:.2f}",
            f"  Hallucination Rate   : {self.our_metrics.hallucination_rate:.1%}",
            f"  Clarification Rate   : {self.our_metrics.clarification_rate:.1%}",
            f"  Avg Docs Retrieved   : {self.our_metrics.avg_docs_retrieved}",
            f"  Failed Runs          : {self.our_metrics.failed_runs}/{self.our_metrics.total_runs}",
        ]

        if self.ragas_scores:
            lines += [
                "",
                "── RAGAS Scores ─────────────────────────────────────────",
            ]
            for k, v in self.ragas_scores.items():
                lines.append(f"  {k:<28}: {v:.3f}")
        elif self.ragas_error:
            lines.append(f"\n  RAGAS: skipped ({self.ragas_error})")

        if self.deepeval_scores:
            lines += [
                "",
                "── DeepEval Scores ──────────────────────────────────────",
            ]
            for k, v in self.deepeval_scores.items():
                lines.append(f"  {k:<28}: {v:.3f}")
        elif self.deepeval_error:
            lines.append(f"\n  DeepEval: skipped ({self.deepeval_error})")

        lines.append("=" * 60)
        return lines

    def print_summary(self) -> None:
        text = "\n".join(self.summary_lines())
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("ascii", "replace").decode("ascii"))

    def save(self) -> Path:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_ts = self.timestamp.replace(":", "-").replace(".", "-")
        path = REPORTS_DIR / f"{safe_ts}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info("eval_report_saved", extra={"path": str(path)})
        return path


def _compute_our_metrics(
    records: List[GoldenRecord], results: List[PipelineResult]
) -> OurMetrics:
    total = len(results)
    if total == 0:
        return OurMetrics(
            router_accuracy=0.0,
            avg_confidence_score=0.0,
            avg_retry_count=0.0,
            hallucination_rate=0.0,
            clarification_rate=0.0,
            avg_docs_retrieved=0,
            failed_runs=0,
            total_runs=0,
        )

    failed = sum(1 for r in results if r.error)
    successful = [r for r in results if not r.error]
    n = len(successful) or 1

    # Router accuracy: check if expected_strategy appears in strategies_used
    router_hits = 0
    for record, result in zip(records, results):
        if result.error:
            continue
        if record.expected_strategy and record.expected_strategy in result.strategies_used:
            router_hits += 1
        elif not record.expected_strategy:
            router_hits += 1  # no expectation = not counted against

    router_accuracy = router_hits / n

    avg_confidence = sum(r.confidence_score for r in successful) / n
    avg_retry = sum(r.retry_count for r in successful) / n
    hallucination_rate = sum(
        1 for r in successful if r.reflection_hallucinations
    ) / n
    clarification_rate = sum(
        1 for r in successful if r.clarification_needed
    ) / n
    avg_docs = int(
        sum(len(r.retrieved_contexts) for r in successful) / n
    )

    return OurMetrics(
        router_accuracy=router_accuracy,
        avg_confidence_score=avg_confidence,
        avg_retry_count=avg_retry,
        hallucination_rate=hallucination_rate,
        clarification_rate=clarification_rate,
        avg_docs_retrieved=avg_docs,
        failed_runs=failed,
        total_runs=total,
    )


def build_report(
    records: List[GoldenRecord],
    results: List[PipelineResult],
    ragas_result=None,
    deepeval_result=None,
) -> EvalReport:
    our_metrics = _compute_our_metrics(records, results)

    per_sample = []
    for record, result in zip(records, results):
        per_sample.append({
            "question": record.question,
            "expected_strategy": record.expected_strategy,
            "strategies_used": result.strategies_used,
            "strategy_correct": (
                record.expected_strategy in result.strategies_used
                if record.expected_strategy and not result.error
                else None
            ),
            "confidence_score": result.confidence_score,
            "confidence_level": result.confidence_level,
            "retry_count": result.retry_count,
            "hallucinations": result.reflection_hallucinations,
            "clarification_needed": result.clarification_needed,
            "answer_preview": result.answer[:200] if result.answer else "",
            "error": result.error,
        })

    return EvalReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_queries=len(records),
        our_metrics=our_metrics,
        ragas_scores=ragas_result.scores if ragas_result and ragas_result.available and not ragas_result.error else {},
        ragas_error=ragas_result.error if ragas_result else "not run",
        deepeval_scores=deepeval_result.scores if deepeval_result and deepeval_result.available and not deepeval_result.error else {},
        deepeval_error=deepeval_result.error if deepeval_result else "not run",
        per_sample=per_sample,
    )
