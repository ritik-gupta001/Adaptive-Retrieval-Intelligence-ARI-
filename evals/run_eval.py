import argparse
import asyncio
import sys

sys.path.append(".")

from app.core.logging import configure_logging

configure_logging()

from evals.golden_dataset import load_golden_dataset
from evals.pipeline_runner import run_pipeline
from evals.report import build_report


async def _run(args: argparse.Namespace) -> None:
    records = load_golden_dataset()
    print(f"Loaded {len(records)} golden queries.")

    if args.dry_run:
        print("Dry run — dataset valid. No LLM calls made.")
        return

    print("Running pipeline on all golden queries (sequential)...")
    questions = [r.question for r in records]
    results = await run_pipeline(questions)

    ragas_result = None
    deepeval_result = None

    if not args.deepeval_only:
        print("Running RAGAS evaluation...")
        from evals.ragas_eval import run_ragas_eval
        ragas_result = run_ragas_eval(records, results)

    if not args.ragas_only:
        print("Running DeepEval evaluation...")
        from evals.deepeval_eval import run_deepeval_eval
        deepeval_result = run_deepeval_eval(records, results)

    report = build_report(records, results, ragas_result, deepeval_result)
    report.print_summary()
    path = report.save()
    print(f"\nReport saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARI evaluation harness")
    parser.add_argument("--ragas-only", action="store_true")
    parser.add_argument("--deepeval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
