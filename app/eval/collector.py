"""EvalCollector — non-intrusive conversation trace collector."""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import Any

from app.hook import HookPoint, hook
from app.pipeline.abc import EventConsumer
from app.schema import AgentCompletion, FinishReason


# ── config ──

def _eval_config() -> dict:
    """Read [eval] section from config.toml."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.toml")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data.get("eval", {})


# ── redaction ──

def _is_sensitive_key(key: str, redact_keys: frozenset) -> bool:
    """Check if a key matches any entry in redact_keys.

    Three-tier matching (case-insensitive):
    1. Full-key equality -- ``api_key`` matches ``api_key`` exactly.
    2. Dotted-segment match -- ``auth.token.value`` matches ``token``.
    3. Compound suffix match -- ``x-api-key`` matches ``api_key`` when both
       are normalised (``_``/``-`` treated the same) and the sensitive key's
       segments form the **trailing** segments of the input key.  This avoids
       false positives such as ``api_key_v2`` or ``token_count``.
    """
    key_lower = key.lower()

    # 1. Full-key exact match
    if key_lower in redact_keys:
        return True

    # 2. Dotted-segment match (only ``.`` as separator)
    if any(seg in redact_keys for seg in key_lower.split(".")):
        return True

    # 3. Compound key: normalise ``_``/``-`` and check trailing segments
    input_segments = re.split(r"[_\-]", key_lower)
    if len(input_segments) > 1:
        for rk in redact_keys:
            rk_lower = rk.lower()
            # Only attempt segment matching for compound sensitive keys
            rk_segments = re.split(r"[_\-]", rk_lower)
            if len(rk_segments) > 1 and len(input_segments) >= len(rk_segments):
                if input_segments[-len(rk_segments):] == rk_segments:
                    return True

    return False


def _redact_value_patterns(text: str, redact_keys: frozenset) -> str:
    """Scan string values for key=value / key:value patterns and redact values."""
    for rk in redact_keys:
        # Capture everything after ``key:`` / ``key=`` up to the next comma,
        # semicolon, or end-of-string so that multi-word values (e.g.
        # ``Authorisation: Bearer sk-abc123``) are fully redacted.
        pattern = re.compile(
            rf'({re.escape(rk)})\s*[:=]\s*([^,;]+)',
            re.IGNORECASE,
        )
        text = pattern.sub(r'\1=***', text)
    return text


def redact(obj: Any, redact_keys: frozenset) -> Any:
    """Two-pass redaction: keys first, then value patterns."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if _is_sensitive_key(k, redact_keys):
                result[k] = "***"
            else:
                result[k] = redact(v, redact_keys)
        return result
    if isinstance(obj, list):
        return [redact(v, redact_keys) for v in obj]
    if isinstance(obj, str) and len(obj) > 2:
        return _redact_value_patterns(obj, redact_keys)
    return obj


# ── sync I/O helpers (called via run_in_executor) ──

def _open_file(path: str):
    return open(path, "a", encoding="utf-8")

def _close_file(f):
    f.close()

def _sync_write(f, line: str) -> None:
    """Write + fsync a line. Called in thread executor."""
    f.write(line)
    f.flush()
    os.fsync(f.fileno())


# ── serialization ──

def _serializable_messages(messages: list) -> list[dict]:
    """Convert Message objects to plain dicts for JSON serialization."""
    result = []
    for m in messages:
        d = {"role": m.role.value if hasattr(m.role, 'value') else str(m.role)}
        if m.content:
            d["content"] = m.content
        if m.name:
            d["name"] = m.name
        if m.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in m.tool_calls
            ]
        if m.injected:
            d["injected"] = True
        result.append(d)
    return result


# ── EvalCollector ──

class EvalCollector(EventConsumer):
    """Collects conversation traces for model evaluation.

    Uses @hook annotations (registered via QueuePipeline.start(ctx))
    to capture lifecycle data without modifying the agent flow.
    Writes one JSONL line per turn via run_in_executor for non-blocking I/O.
    """

    def __init__(self):
        cfg = _eval_config()
        self._enabled = cfg.get("enabled", True)
        self._output_dir = cfg.get("output_dir", "data/eval")
        self._redact_keys = frozenset(
            k.lower() for k in cfg.get("redact_keys", [])
        )

        self._session_id: str = ""
        self._file: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._turn_count: int = 0
        self._msg_snapshot: int = 0
        self._turn_buffer: dict | None = None
        self._pending_tools: list[dict] = []

    # ── EventConsumer ──

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        if self._file is not None:
            f = self._file
            self._file = None
            if self._loop is not None:
                await self._loop.run_in_executor(None, _close_file, f)

    async def consume(self, event: Any) -> None:
        if not self._enabled:
            return
        if isinstance(event, AgentCompletion):
            self._on_llm_output(event)

    # ── hooks ──

    @hook(HookPoint.FLOW_START)
    async def _on_flow_start(self, ctx) -> None:
        if not self._enabled:
            return
        self._session_id = uuid.uuid4().hex[:12]
        self._loop = asyncio.get_running_loop()
        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, f"{self._session_id}.jsonl")
        self._file = await self._loop.run_in_executor(None, _open_file, path)
        self._write_session_start(ctx)

    @hook(HookPoint.NODE_EXEC_BEFORE)
    def _on_llm_before(self, ctx, node) -> None:
        if not self._enabled:
            return
        delta = self._capture_messages_delta(ctx)
        user_msg = ""
        for m in reversed(delta):
            if m.get("role") == "user" and not m.get("injected"):
                user_msg = m.get("content", "")
                break
        self._turn_buffer = {
            "type": "turn",
            "turn": self._turn_count,
            "user": user_msg,
            "llm_input_delta": delta,
        }
        self._pending_tools = []

    @hook(HookPoint.TOOL_BEFORE)
    async def _on_tool_before(self, ctx, tool_call, tool) -> None:
        if not self._enabled or self._turn_buffer is None:
            return
        self._pending_tools.append({
            "name": tool_call.function.name,
            "arguments": redact(
                tool_call.function.arguments_dict, self._redact_keys,
            ),
            "id": tool_call.id,
        })

    @hook(HookPoint.TOOL_AFTER)
    async def _on_tool_after(self, ctx, tool_call, tool, result) -> None:
        if not self._enabled or self._turn_buffer is None:
            return
        for t in self._pending_tools:
            if t["id"] == tool_call.id:
                t["result"] = redact(
                    {
                        "content": result.content,
                        "success": getattr(result, "success", None),
                    },
                    self._redact_keys,
                )
                break

    # ── internal ──

    def _on_llm_output(self, event: AgentCompletion) -> None:
        buf = self._turn_buffer
        if buf is None:
            return
        if event.finish_reason and event.finish_reason != FinishReason.NONE:
            buf["llm_output"] = {
                "content": event.full_content,
                "tool_calls": [
                    {"name": tc.function.name, "id": tc.id}
                    for tc in (event.full_tool_calls or [])
                ],
                "finish_reason": event.finish_reason.value,
            }
            buf["tool_calls"] = self._pending_tools
            self._turn_count += 1
            self._flush_turn(buf)
            self._turn_buffer = None
            self._pending_tools = []

    def _write_session_start(self, ctx) -> None:
        msgs = ctx.memory.messages if hasattr(ctx, 'memory') else []
        initial = _serializable_messages(msgs)
        self._msg_snapshot = len(msgs)
        self._write_line({
            "type": "session_start",
            "session_id": self._session_id,
            "initial_messages": initial,
        })

    def _capture_messages_delta(self, ctx) -> list[dict]:
        msgs = ctx.memory.messages if hasattr(ctx, 'memory') else []
        delta = msgs[self._msg_snapshot:]
        self._msg_snapshot = len(msgs)
        return _serializable_messages(delta)

    def _flush_turn(self, buf: dict) -> None:
        self._write_line(buf)

    def _write_line(self, obj: dict) -> None:
        if self._file is None or self._loop is None:
            return
        line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
        self._loop.run_in_executor(None, _sync_write, self._file, line)
