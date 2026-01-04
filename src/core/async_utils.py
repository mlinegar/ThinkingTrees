"""
Async Utilities for OPS Framework.

This module provides helper functions for async operations, particularly
around proper task cleanup to prevent orphaned tasks from piling up.
"""

import asyncio
import logging
from typing import Any, Iterable, List, Coroutine

logger = logging.getLogger(__name__)


async def gather_with_cleanup(
    coros: Iterable[Coroutine[Any, Any, Any]],
    return_exceptions: bool = True,
) -> List[Any]:
    """
    Gather coroutines with proper cleanup on cancellation.

    Unlike asyncio.gather, this function ensures all tasks are cancelled
    if the gather is cancelled (e.g., via Ctrl+C or timeout).

    Args:
        coros: Iterable of coroutines to run concurrently
        return_exceptions: If True, exceptions are returned as results.
                          If False, first exception is raised.

    Returns:
        List of results (or exceptions if return_exceptions=True)

    Example:
        # Safe gather that cleans up on cancellation
        results = await gather_with_cleanup([
            summarize_leaf(chunk) for chunk in chunks
        ])
    """
    # Convert coroutines to tasks
    tasks = [asyncio.ensure_future(c) for c in coros]

    if not tasks:
        return []

    try:
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
    except asyncio.CancelledError:
        # Cancel all pending tasks
        cancelled_count = 0
        for task in tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1

        if cancelled_count > 0:
            logger.debug(f"Cancelled {cancelled_count} pending tasks due to cancellation")
            # Wait briefly for cancellation to propagate
            await asyncio.gather(*tasks, return_exceptions=True)

        # Re-raise the cancellation
        raise


async def cancel_tasks(tasks: Iterable[asyncio.Task], timeout: float = 5.0) -> int:
    """
    Cancel a collection of tasks and wait for them to complete.

    Args:
        tasks: Tasks to cancel
        timeout: Maximum time to wait for cancellation

    Returns:
        Number of tasks that were successfully cancelled
    """
    task_list = list(tasks)
    if not task_list:
        return 0

    # Cancel all tasks
    cancelled = 0
    for task in task_list:
        if not task.done():
            task.cancel()
            cancelled += 1

    if cancelled == 0:
        return 0

    # Wait for cancellation with timeout
    try:
        await asyncio.wait_for(
            asyncio.gather(*task_list, return_exceptions=True),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        remaining = sum(1 for t in task_list if not t.done())
        logger.warning(
            f"Timeout ({timeout}s) waiting for task cancellation. "
            f"{remaining}/{len(task_list)} tasks may still be running."
        )

    return cancelled
