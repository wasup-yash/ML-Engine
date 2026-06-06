import asyncio
import inspect
from typing import Any, Callable, List, Tuple


class AsyncBatcher:
    def __init__(
        self,
        infer_fn: Callable[[List[Any]], Any],
        max_batch_size: int = 32,
        max_wait_ms: int = 20,
    ) -> None:
        self.infer_fn = infer_fn
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self._queue: "asyncio.Queue[Tuple[Any, asyncio.Future]]" = asyncio.Queue()
        self._runner_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._runner_task is None or self._runner_task.done():
                self._runner_task = asyncio.create_task(self._run(), name="async_batcher_runner")

    async def submit(self, item: Any) -> Any:
        await self.start()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put((item, future))
        return await future

    async def _call_infer(self, batch_inputs: List[Any]) -> List[Any]:
        if inspect.iscoroutinefunction(self.infer_fn):
            result = await self.infer_fn(batch_inputs)
        else:
            maybe_awaitable = await asyncio.to_thread(self.infer_fn, batch_inputs)
            if inspect.isawaitable(maybe_awaitable):
                result = await maybe_awaitable
            else:
                result = maybe_awaitable

        if isinstance(result, list):
            return result
        if len(batch_inputs) == 1:
            return [result]
        raise ValueError("Batch inference function must return a list for batched input")

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first_item, first_future = await self._queue.get()
            batch = [(first_item, first_future)]
            deadline = loop.time() + self.max_wait_s

            while len(batch) < self.max_batch_size:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    item, future = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    batch.append((item, future))
                except asyncio.TimeoutError:
                    break

            inputs = [request for request, _ in batch]
            futures = [future for _, future in batch]

            try:
                outputs = await self._call_infer(inputs)
                if len(outputs) != len(batch):
                    raise ValueError(
                        f"Batch output size mismatch: got {len(outputs)} for {len(batch)} requests"
                    )
                for future, output in zip(futures, outputs):
                    if not future.done():
                        future.set_result(output)
            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
