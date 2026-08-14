
import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _dummy_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
    _dummy_vertex.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = _dummy_vertex

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.core.logging import get_logger
from evals.golden_dataset import GoldenRecord
from evals.pipeline_runner import PipelineResult

logger = get_logger(__name__)


@dataclass
class RagasEvalResult:
    available: bool = False
    scores: Dict[str, float] = field(default_factory=dict)
    per_sample: List[Dict] = field(default_factory=list)
    error: Optional[str] = None


def _build_ragas_dataset(
    records: List[GoldenRecord], results: List[PipelineResult]
):
    """Build a RAGAS EvaluationDataset from golden records + pipeline outputs."""
    from ragas import EvaluationDataset, SingleTurnSample

    samples = []
    for record, result in zip(records, results):
        if result.error:
            continue  # skip failed pipeline runs
        samples.append(
            SingleTurnSample(
                user_input=record.question,
                response=result.answer,
                retrieved_contexts=result.retrieved_contexts
                if result.retrieved_contexts
                else ["[no context retrieved]"],
                reference=record.ground_truth,
            )
        )
    return EvaluationDataset(samples=samples)


def _make_ragas_llm_and_embeddings():
    """Build RAGAS-compatible LLM + Embeddings wrappers using the configured provider.

    Checks configured settings.llm_provider first, then falls back to available provider keys.
    """
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from app.config.settings import settings
    from langchain_community.embeddings import FakeEmbeddings

    provider = settings.llm_provider

    # Anthropic
    if (provider == "anthropic" or not settings.openai_api_key) and settings.anthropic_api_key:
        try:
            from langchain_anthropic import ChatAnthropic
            llm = LangchainLLMWrapper(ChatAnthropic(
                model=settings.llm_model if provider == "anthropic" else "claude-3-5-sonnet-20241022",
                api_key=settings.anthropic_api_key,
            ))
            embeddings = LangchainEmbeddingsWrapper(FakeEmbeddings(size=1536))
            logger.info("ragas_using_anthropic_scorer")
            return llm, embeddings
        except ImportError:
            pass

    # OpenAI
    if settings.openai_api_key:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        llm = LangchainLLMWrapper(ChatOpenAI(
            model=settings.llm_model if provider == "openai" else "gpt-4o-mini",
            api_key=settings.openai_api_key,
        ))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        ))
        logger.info("ragas_using_openai_scorer")
        return llm, embeddings

    # Groq fallback
    if settings.groq_api_key:
        try:
            from langchain_groq import ChatGroq
            llm = LangchainLLMWrapper(ChatGroq(
                model=settings.llm_model if provider == "groq" else "llama-3.3-70b-versatile",
                api_key=settings.groq_api_key,
            ))
            embeddings = LangchainEmbeddingsWrapper(FakeEmbeddings(size=1536))
            logger.info("ragas_using_groq_scorer_with_fake_embeddings")
            return llm, embeddings
        except ImportError:
            pass

    # Google Gemini fallback
    if settings.google_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
                model=settings.llm_model if provider == "gemini" else "gemini-1.5-flash",
                google_api_key=settings.google_api_key,
            ))
            embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=settings.google_api_key,
            ))
            logger.info("ragas_using_gemini_scorer")
            return llm, embeddings
        except ImportError:
            pass

    raise ImportError(
        "No supported LLM provider available for RAGAS scoring. "
        "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY."
    )


def run_ragas_eval(
    records: List[GoldenRecord],
    results: List[PipelineResult],
) -> RagasEvalResult:
    """
    Run RAGAS evaluation. Returns RagasEvalResult with available=False
    if ragas is not installed or if the eval run itself fails.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        logger.warning("ragas_not_installed", extra={"error": str(exc)})
        return RagasEvalResult(
            available=False, error=f"ragas not installed: {exc}"
        )

    try:
        llm, embeddings = _make_ragas_llm_and_embeddings()
    except ImportError as exc:
        logger.warning("ragas_no_scorer_llm", extra={"error": str(exc)})
        return RagasEvalResult(
            available=False, error=f"No LLM available for RAGAS scoring: {exc}"
        )

    try:
        dataset = _build_ragas_dataset(records, results)
        if not dataset.samples:
            return RagasEvalResult(
                available=True,
                error="All pipeline runs failed — no samples to evaluate",
            )

        metrics = [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
        ]

        eval_result = evaluate(dataset=dataset, metrics=metrics)
        df = eval_result.to_pandas()

        aggregate_scores = {}
        for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if col in df.columns:
                aggregate_scores[col] = float(df[col].mean())

        per_sample = df.to_dict(orient="records")

        logger.info("ragas_eval_completed", extra={"scores": aggregate_scores})
        return RagasEvalResult(
            available=True, scores=aggregate_scores, per_sample=per_sample
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("ragas_eval_failed", extra={"error": str(exc)})
        return RagasEvalResult(available=True, error=str(exc))
