from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel

# 类型别名：type X = Y  Y：Callable[[a,b,...],[x]] -> 接受a,b,...参数，返回x参数的可调用对象
# Callable：表示可调用对象
# Awaitable： 当返回值是用await调用的对象，x：Awaitable[...]
type EventHandler = Callable[[BaseModel], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[EventHandler] = []

    # 注册一个事件处理函数
    def subscribe(self, handler: EventHandler) -> None:
        self._subscribers.append(handler)

    # 按注册顺序依次调用所有订阅者
    async def publish(self, event: BaseModel) -> None:
        for handler in self._subscribers:
            await handler(event)
