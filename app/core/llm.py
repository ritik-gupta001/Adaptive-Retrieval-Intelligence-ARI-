import asyncio
import json
import re
import time
from functools import lru_cache

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from app.config.settings import settings
from app.core.exceptions import LLMCallError
from app.core.logging import get_logger

logger = get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> Dict[str, str]:
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise LLMCallError(f"Prompt file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def render(template: str, **kwargs) -> str:
    """Single-pass, safe template substitution that handles literal braces in templates.

    The prompt YAML files contain JSON examples with literal { and } characters that
    must NOT be treated as format-string placeholders. Strategy:
    1. Escape all literal {{ and }} that are already doubled (they stay doubled).
    2. For each kwarg key, replace the exact pattern {key} with the value using
       a simple str.replace (not format_map) — this is safe and unambiguous
       because we process each key exactly once and values are str()-converted.

    This avoids the re-substitution collision risk (a value containing {other_key}
    would only be processed during its own replace call, not in a second pass) while
    also not breaking on JSON example text in prompts.
    """
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _strip_json_fences(raw: str) -> str:
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return cleaned


def _parse_json(raw: str) -> Dict[str, Any]:
    cleaned = _strip_json_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


_client_cache: Dict[str, Any] = {}


async def _get_async_client_for_provider(provider: str):
    if provider in _client_cache:
        return _client_cache[provider]

    timeout = settings.llm_client_timeout
    max_retries = settings.llm_client_max_retries

    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=timeout, max_retries=max_retries)
    elif provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=timeout, max_retries=max_retries)
    elif provider == "groq":
        from groq import AsyncGroq
        client = AsyncGroq(api_key=settings.groq_api_key, timeout=timeout, max_retries=max_retries)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        client = ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            model=settings.gemini_fallback_model,
            request_timeout=timeout,
        )
    else:
        raise LLMCallError(f"Unsupported provider for async client: {provider}")

    _client_cache[provider] = client
    return client


async def _call_single_provider(provider: str, system: str, user: str, max_tokens: int = 1024) -> str:
    client = await _get_async_client_for_provider(provider)
    # Primary provider uses the configured model; fallbacks use per-provider fallback models
    fallback_models = {
        "openai": settings.openai_fallback_model,
        "groq": settings.groq_fallback_model,
        "anthropic": settings.anthropic_fallback_model,
        "gemini": settings.gemini_fallback_model,
    }
    model = settings.llm_model if provider == settings.llm_provider else fallback_models.get(provider, settings.llm_model)

    if provider == "anthropic":
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))

    if provider in ("openai", "groq"):
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    if provider == "gemini":
        if hasattr(client, "ainvoke"):
            res = await client.ainvoke(f"{system}\n\n{user}")
        else:
            res = await asyncio.to_thread(client.invoke, f"{system}\n\n{user}")
        return res.content or ""

    raise LLMCallError(f"Unsupported provider: {provider}")


async def _call_provider(system: str, user: str, max_tokens: int = 1024) -> str:
    return await _call_single_provider(settings.llm_provider, system, user, max_tokens=max_tokens)


async def call_llm(
    prompt_name: str,
    *,
    max_tokens: int = 1024,
    max_retries: int = 1,
    per_attempt_timeout: float = 15.0,
    **kwargs,
) -> str:
    """Render prompt_name.yaml with kwargs, call LLM with automatic provider fallback.

    per_attempt_timeout: seconds before a single provider call is cancelled and
    the next attempt/provider is tried. Prevents one hung provider from stalling
    the entire retry chain (which can compound badly with the graph retry loop).
    """
    spec = load_prompt(prompt_name)
    system = render(spec["system"], **kwargs)
    user = render(spec["user"], **kwargs)

    primary = settings.llm_provider
    providers_to_try = [primary]
    if settings.groq_api_key and "groq" not in providers_to_try:
        providers_to_try.append("groq")
    if settings.openai_api_key and "openai" not in providers_to_try:
        providers_to_try.append("openai")
    if settings.google_api_key and "gemini" not in providers_to_try:
        providers_to_try.append("gemini")
    if settings.anthropic_api_key and "anthropic" not in providers_to_try:
        providers_to_try.append("anthropic")

    last_error: Optional[Exception] = None
    for provider in providers_to_try:
        for attempt in range(max_retries + 1):
            start = time.monotonic()
            try:
                coro = (
                    _call_provider(system, user, max_tokens=max_tokens)
                    if provider == primary
                    else _call_single_provider(provider, system, user, max_tokens=max_tokens)
                )
                raw = await asyncio.wait_for(coro, timeout=per_attempt_timeout)
                latency = round(time.monotonic() - start, 3)
                logger.info(
                    "llm_call_succeeded",
                    extra={
                        "prompt": prompt_name,
                        "provider": provider,
                        "attempt": attempt,
                        "latency_seconds": latency,
                    },
                )
                return raw
            except (asyncio.TimeoutError, TimeoutError):
                last_error = TimeoutError(
                    f"Provider '{provider}' timed out after {per_attempt_timeout}s"
                )
                logger.warning(
                    "llm_call_timeout",
                    extra={
                        "prompt": prompt_name,
                        "provider": provider,
                        "attempt": attempt,
                        "timeout_seconds": per_attempt_timeout,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "llm_call_failed_attempt",
                    extra={
                        "prompt": prompt_name,
                        "provider": provider,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )
            if attempt < max_retries:
                await asyncio.sleep(0.2 * (attempt + 1))

    raise LLMCallError(
        f"LLM call '{prompt_name}' failed across providers {providers_to_try}: {last_error}"
    )


async def call_llm_json(prompt_name: str, *, max_tokens: int = 1024, **kwargs) -> Dict[str, Any]:
    """Call the LLM and parse+return JSON. Raises LLMCallError if the model
    never returns parseable JSON within the retry budget."""
    raw = await call_llm(prompt_name, max_tokens=max_tokens, **kwargs)
    try:
        return _parse_json(raw)
    except json.JSONDecodeError as exc:
        logger.error(
            "llm_json_parse_failed",
            extra={"prompt": prompt_name, "raw_output": raw[:500]},
        )
        raise LLMCallError(f"LLM '{prompt_name}' did not return valid JSON: {exc}") from exc
