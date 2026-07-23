from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from kama_claude.core.bus.events import RunFinishedEvent, RunStartedEvent
from kama_claude.core.config import KamaConfig
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus, EventHandler
from kama_claude.core.events.writer import EventWriter
from kama_claude.core.llm.base import LLMProvider
from kama_claude.core.llm.provider import AnthropicProvider
from kama_claude.core.loop import AgentLoop
from kama_claude.core.runs import RUNS_DIR, new_run_id
from kama_claude.core.tools.builtin.read_file import ReadFileTool
from kama_claude.core.tools.registry import ToolRegistry


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AgentRunner:
    # 组装所有运行时依赖，准备执行一次完整的 agent run
    def __init__(
        self,
        config: KamaConfig,
        *,
        provider: LLMProvider | None = None,
        extra_handlers: list[EventHandler] | None = None,
        runs_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._extra_handlers: list[EventHandler] = extra_handlers or []
        self._runs_dir = runs_dir or RUNS_DIR

    # 执行一次完整的 agent run：生成 run_id、接线事件总线、驱动 AgentLoop
    async def run(self, goal: str) -> None:
        # 生成唯一 run_id 并创建对应的运行目录（如 runs/20260722-163000-abc123/）
        run_id = new_run_id()
        run_path = self._runs_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        # 创建事件总线，订阅所有监听者
        bus = EventBus() # 初始化订阅者列表：[]
        for h in self._extra_handlers:
            bus.subscribe(h) # 每个 handler 加到订阅者列表里

        # 获取 LLM 提供方，注册内置工具，组装循环控制器
        provider = self._provider or AnthropicProvider(self._config.llm.default_model)
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        loop = AgentLoop(provider, registry, bus)

        # 创建执行上下文：记录 run_id、目标 goal、最大步数限制，自动插入第一条 user message
        context = ExecutionContext(
            run_id=run_id,
            goal=goal,
            max_steps=self._config.agent.max_steps,
        )

        # 打开事件持久化文件写入器（runs/<run_id>/events.jsonl），订阅事件总线
        async with EventWriter(run_path / "events.jsonl") as writer:
            writer.subscribe(bus)
            # 发布 run.started 事件，标记运行开始
            await bus.publish(RunStartedEvent(run_id=run_id, goal=goal, ts=_now()))

            cancelled = False
            try:
                # 进入 plan→act→observe 循环（由 AgentLoop 驱动）
                await loop.run(context)
            except asyncio.CancelledError:
                # 捕获取消信号（如 Ctrl+C），将运行标记为 cancelled
                cancelled = True
                if not context.is_done():
                    context.mark_failed("cancelled")

            # 发布 run.finished 事件：状态、原因、总步数
            await bus.publish(
                RunFinishedEvent(
                    run_id=run_id,
                    status=context.status,
                    reason=context.reason,
                    steps=context.step,
                    ts=_now(),
                )
            )

        # 如果被取消，将 CancelledError 继续向上传播给调用方
        if cancelled:
            raise asyncio.CancelledError()
