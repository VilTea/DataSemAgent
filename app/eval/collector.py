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
from app.schema import FinishReason


# ── config ──

from app.config import config


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


# ── sync I/O helpers ──

def _write_file(path: str, events: list[dict]) -> None:
    """Write all buffered events as JSONL."""
    with open(path, "a", encoding="utf-8") as f:
        for obj in events:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ── serialization ──

def _serializable_messages(messages: list) -> list[dict]:
    """Convert Message objects to plain dicts, keeping ALL fields."""
    return [m.model_dump(mode="json", exclude_none=True) for m in messages]


# ── EvalCollector ──

class EvalCollector(EventConsumer):
    """Collects conversation traces for model evaluation.

    Events are buffered in-memory in hook firing order (guaranteeing
    chronological ordering) and flushed to a JSONL file at stop/FLOW_END.

    Each event type gets its own line:

    session_start — session metadata, system prompt, initial messages
    turn_start    — new turn / user question
    llm_input     — messages delta sent to the LLM
    llm_output    — LLM response (reasoning, content, tool_calls, finish_reason)
    tool_call     — single tool execution (name, arguments, result)

    Turn boundary = STOP.  A single user question may produce multiple
    Agent→Tool→Agent cycles within the same turn.
    """

    def __init__(self):
        cfg = config.eval["default"]
        self._enabled = cfg.enabled
        self._output_dir = cfg.output_dir
        self._redact_keys = frozenset(k.lower() for k in cfg.redact_keys)

        self._session_id: str = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._turn_count: int = 0
        self._turn_started: bool = False
        self._msg_snapshot: int = 0
        self._pending_tools: dict[str, dict] = {}  # tool_call_id → {name, arguments}
        self._metadata: dict = {}
        self._events: list[dict] = []  # in-memory event buffer

    def set_metadata(self, meta: dict) -> None:
        """Attach task-level metadata (task_id, question, etc.) to the trace."""
        self._metadata = dict(meta)

    # ── EventConsumer ──

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        await self._flush()

    async def consume(self, event: Any) -> None:
        pass  # all capture is done via hooks, not pipeline events

    # ── hooks ──

    @hook(HookPoint.FLOW_START)
    async def _on_flow_start(self, ctx) -> None:
        if not self._enabled:
            return
        self._session_id = uuid.uuid4().hex[:12]
        self._loop = asyncio.get_running_loop()

    @hook(HookPoint.FLOW_END)
    async def _on_flow_end(self, ctx) -> None:
        if not self._enabled:
            return
        await self._flush()

    @hook(HookPoint.NODE_EXEC_BEFORE)
    async def _on_llm_before(self, ctx, node) -> None:
        if not self._enabled:
            return

        # Capture delta BEFORE writing session_start — _build_session_start
        # also reads ctx.memory.messages and must not advance the snapshot.
        delta = self._capture_messages_delta(ctx)
        if not delta:
            return

        # Lazy session_start: system_prompt is available from node now.
        if self._turn_count == 0 and not self._turn_started:
            self._events.append(self._build_session_start(ctx, node))

        # First LLM call of a new turn → write turn_start before llm_input.
        if not self._turn_started:
            user_msg = ""
            for m in delta:
                if m.get("role") == "user" and not m.get("injected"):
                    user_msg = m.get("content", "")
            self._events.append({
                "type": "turn_start",
                "turn": self._turn_count,
                "user": user_msg,
            })
            self._turn_started = True

        self._events.append({
            "type": "llm_input",
            "turn": self._turn_count,
            "messages": delta,
        })

    @hook(HookPoint.TOOL_BEFORE)
    async def _on_tool_before(self, ctx, tool_call, tool) -> None:
        if not self._enabled:
            return
        self._pending_tools[tool_call.id] = {
            "name": tool_call.function.name,
            "arguments": redact(
                tool_call.function.arguments_dict, self._redact_keys,
            ),
        }

    @hook(HookPoint.TOOL_AFTER)
    async def _on_tool_after(self, ctx, tool_call, tool, result) -> None:
        if not self._enabled:
            return
        tc = self._pending_tools.pop(tool_call.id, None)
        if tc is None:
            return
        dumped = result.model_dump(mode="json", exclude_none=True)
        dumped["success"] = result.is_success()
        tc["result"] = redact(dumped, self._redact_keys)
        self._events.append({
            "type": "tool_call",
            "turn": self._turn_count,
            "name": tc["name"],
            "arguments": tc.get("arguments", {}),
            "result": tc.get("result", {}),
        })

    @hook(HookPoint.NODE_EXEC_AFTER)
    async def _on_llm_after(self, ctx, node, reason) -> None:
        """Capture LLM output from the last assistant message.

        Uses NODE_EXEC_AFTER (synchronous within agent node exec) so the
        llm_output lands in the event buffer before subsequent tool_call
        events from the same cycle.
        """
        if not self._enabled:
            return
        finish = reason
        if finish is None or finish == FinishReason.NONE:
            return

        # Reconstruct from the last message in memory.
        msgs = ctx.memory.messages if hasattr(ctx, 'memory') else []
        last_msg = msgs[-1] if msgs else None
        if last_msg is None or last_msg.role != "assistant":
            return

        self._events.append({
            "type": "llm_output",
            "turn": self._turn_count,
            "content": getattr(last_msg, 'content', '') or '',
            "reasoning_content": getattr(last_msg, 'reasoning_content', '') or '',
            "tool_calls": [
                {
                    "name": tc.function.name,
                    "id": tc.id,
                    "arguments": tc.function.arguments,
                }
                for tc in (getattr(last_msg, 'tool_calls', None) or [])
            ],
            "finish_reason": finish.value,
        })
        if finish == FinishReason.STOP:
            self._turn_count += 1
            self._turn_started = False

    # ── internal ──

    def _build_session_start(self, ctx, node) -> dict:
        msgs = ctx.memory.messages if hasattr(ctx, 'memory') else []
        initial = _serializable_messages(msgs)
        sys_prompt = getattr(node, 'system_prompt', None) or ""
        # Strip system-prompt content from initial_messages (stored top-level).
        for m in initial:
            if m.get("role") == "system":
                m.pop("content", None)
        header: dict = {
            "type": "session_start",
            "session_id": self._session_id,
            "system_prompt": sys_prompt,
            "initial_messages": initial,
        }
        if self._metadata:
            header["metadata"] = self._metadata
        return header

    def _capture_messages_delta(self, ctx) -> list[dict]:
        msgs = ctx.memory.messages if hasattr(ctx, 'memory') else []
        delta = msgs[self._msg_snapshot:]
        self._msg_snapshot = len(msgs)
        return _serializable_messages(delta)

    async def _flush(self) -> None:
        if not self._events:
            return
        events = self._events
        self._events = []
        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, f"{self._session_id}.jsonl")
        if self._loop is not None:
            await self._loop.run_in_executor(None, _write_file, path, events)
