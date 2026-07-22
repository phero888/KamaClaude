from __future__ import annotations

import asyncio
import json
import sys
import time

import kama_claude
from kama_claude.core.bus.commands import PongResult
from kama_claude.core.bus.envelope import JsonRpcError, JsonRpcSuccess
from kama_claude.core.config import KamaConfig


# 同步入口：运行 ping 协程，连接失败时打印错误并退出
def cmd_ping(config: KamaConfig) -> None:
    try:
        #启动事件循环，执行真正的 _ping 协程
        # 异步：复杂但后续扩展方便
        #同步：socket
        asyncio.run(_ping(config))
    except (ConnectionRefusedError, OSError):
        print(f"error: core not running ({config.host}:{config.port})", file=sys.stderr)
        sys.exit(1)


# 向 core 守护进程发送 ping 请求，打印 pong 响应及延迟
async def _ping(config: KamaConfig) -> None:
    t0 = time.monotonic()
    # 异步：使用 asyncio.open_connection 建立 TCP 连接
    reader, writer = await asyncio.open_connection(config.host, config.port)

    req = {
        "jsonrpc": "2.0",
        "id": "cli-1",
        "method": "core.ping",
        "params": {"client": f"cli/{kama_claude.__version__}"},
    }
    # \n:NDJSON格式，一行一个json对象，方便解析
    writer.write((json.dumps(req) + "\n").encode())
    # drain:将缓存中的数据发送出去
    await writer.drain()

    # 异步：等待并读取一行响应，设置 10 秒超时
    line = await asyncio.wait_for(reader.readline(), timeout=10.0)
    latency_ms = int((time.monotonic() - t0) * 1000)

    # 异步：关闭连接
    writer.close()
    await writer.wait_closed()

    raw = json.loads(line)
    if "error" in raw:
        err = JsonRpcError.model_validate(raw)
        print(f"error: {err.error.code} {err.error.message}", file=sys.stderr)
        sys.exit(1)

    # 两步反序列化：先用 model_validate 校验信封结构，再拿 result 字段校验业务数据。Pydantic 的 model_validate
    resp = JsonRpcSuccess.model_validate(raw)
    result = PongResult.model_validate(resp.result)
    print(f"pong server={result.server_version} uptime={result.uptime_ms}ms latency={latency_ms}ms")
