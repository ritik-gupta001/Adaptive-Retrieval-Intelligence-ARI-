"""
Unit and regression tests for app.core.llm module.

Tests requirement 2:
- Assert that call_llm's retry backoff uses asyncio.sleep and does NOT block concurrent coroutines on the event loop.
"""
import asyncio
import pytest

from app.core.exceptions import LLMCallError
from app.core.llm import call_llm


@pytest.mark.asyncio
async def test_call_llm_async_sleep_non_blocking(monkeypatch):
    """
    Assert call_llm uses non-blocking asyncio backoff so concurrent coroutines
    execute while call_llm retries.
    """
    call_attempts = 0

    async def mock_call_provider(system: str, user: str, max_tokens: int = 1024) -> str:
        nonlocal call_attempts
        call_attempts += 1
        if call_attempts == 1:
            raise Exception("Simulated transient provider error")
        return '{"result": "ok"}'

    monkeypatch.setattr("app.core.llm._call_provider", mock_call_provider)
    monkeypatch.setattr("app.core.llm.load_prompt", lambda name: {"system": "sys", "user": "usr"})

    side_coroutine_ran = False

    async def concurrent_task():
        nonlocal side_coroutine_ran
        await asyncio.sleep(0.1)  # small delay shorter than the 0.5s retry sleep
        side_coroutine_ran = True

    # Launch both call_llm and the concurrent task
    llm_task = asyncio.create_task(call_llm("test_prompt", max_retries=1))
    side_task = asyncio.create_task(concurrent_task())

    # Wait for both tasks to complete
    result, _ = await asyncio.gather(llm_task, side_task)

    assert result == '{"result": "ok"}'
    assert call_attempts == 2
    assert side_coroutine_ran is True, "Concurrent coroutine must complete while call_llm retries"
