import json
import re

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from app.pipeline.abc import EventConsumer
from app.schema import AgentCompletion, FinishReason, Message, Role

_RESULT_PREVIEW_LINES = 20


class RichConsumer(EventConsumer):
    """Stream agent output to a Rich console.

    Text and think content stream inline.  Tool-call panels are rendered
    once, when the LLM response completes (finish_reason == TOOL_CALLS),
    so the full arguments are available.  Tool results use a distinct
    bordered panel to stand apart from think content.
    """

    def __init__(self, console: Console):
        self._console = console
        self._think_open = False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self._flush_think()

    # ------------------------------------------------------------------ #
    #  consume
    # ------------------------------------------------------------------ #

    async def consume(self, event) -> None:
        if isinstance(event, AgentCompletion):
            self._handle_completion(event)
        elif isinstance(event, Message) and event.role == Role.TOOL:
            self._handle_tool_message(event)

    # ------------------------------------------------------------------ #
    #  AgentCompletion — streaming text + deferred tool panels
    # ------------------------------------------------------------------ #

    def _handle_completion(self, event: AgentCompletion) -> None:
        # --- reasoning (think) — dim italic, inline ---
        if event.reasoning_content:
            if not self._think_open:
                self._think_open = True
            self._console.print(event.reasoning_content, end="", style="dim italic")

        # --- text content — normal, inline ---
        if event.content:
            self._flush_think()
            self._console.print(event.content, end="")

        # --- tool calls — render ONCE when streaming is done ---
        if (event.finish_reason == FinishReason.TOOL_CALLS
                and event.full_tool_calls):
            self._flush_think()
            self._console.print()
            for tc in event.full_tool_calls:
                if tc.function.name:
                    self._render_tool_call(tc)

        # --- finish ---
        if event.finish_reason and event.finish_reason != FinishReason.NONE:
            self._console.print()
            self._console.print(Rule(style="dim"))
            self._console.file.flush()

    # ------------------------------------------------------------------ #
    #  Message (TOOL result) — distinct block
    # ------------------------------------------------------------------ #

    def _handle_tool_message(self, msg: Message) -> None:
        content = msg.content or ""
        tool_name = getattr(msg, "name", "") or ""
        body = self._format_result(tool_name, content)
        if body is None:
            return
        self._console.print(Panel(body, border_style="green", padding=(0, 1)))

    # ------------------------------------------------------------------ #
    #  render tool call
    # ------------------------------------------------------------------ #

    def _render_tool_call(self, tc) -> None:
        name = tc.function.name
        args_raw = tc.function.arguments or ""
        preview = self._preview_args(name, args_raw)

        body = Text()
        body.append(f"  {name}", style="bold cyan")
        if preview:
            body.append("\n")
            body.append(preview, style="dim")

        self._console.print(Panel(body, border_style="cyan", padding=(0, 1)))

    def _preview_args(self, tool_name: str, args: str) -> str:
        if not args:
            return ""
        try:
            d = json.loads(args)
        except Exception:
            return args[:300]
        if tool_name == "sql_exec":
            return d.get("sql", "")[:500]
        if tool_name in ("entity_graph", "metric_lineage"):
            return d.get("query", "")[:500]
        if tool_name == "activate_skill":
            return f"name={d.get('name', '')}"[:200]
        items = list(d.items())[:3]
        return ", ".join(f"{k}={str(v)[:60]}" for k, v in items)[:200]

    # ------------------------------------------------------------------ #
    #  format tool result
    # ------------------------------------------------------------------ #

    def _format_result(self, tool_name: str, content: str) -> Text | None:
        if not content.strip():
            return None

        if tool_name == "sql_exec":
            return self._format_sql_result(content)
        if tool_name in ("entity_graph", "metric_lineage"):
            return self._format_graph_result(content)
        if tool_name == "activate_skill":
            first = _first_line(content)
            if first:
                return Text(f"  {first[:300]}", style="dim")
            return None
        first = _first_line(content)
        return Text(f"  {first[:300]}", style="dim") if first else None

    def _format_sql_result(self, content: str) -> Text:
        m = re.search(r"\*\*Total rows\*\*:\s*(\d+)", content)
        total = m.group(1) if m else None
        m2 = re.search(r"\*\*Returned\*\*:\s*(\d+)", content)
        returned = m2.group(1) if m2 else total

        lines = content.strip().split("\n")
        header: list[str] = []
        rows: list[list[str]] = []
        in_table = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and "---" not in stripped:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if not in_table:
                    header = cells
                    in_table = True
                else:
                    if len(cells) == len(header):
                        rows.append(cells)

        body = Text()
        if header and rows:
            label = f"  {returned}/{total} rows" if total else f"  {len(rows)} rows"
            body.append(label, style="dim")
            body.append("\n")
            table = Table(show_header=True, header_style="bold", show_lines=False,
                          padding=(0, 1), collapse_padding=True, box=None)
            for h in header:
                table.add_column(h, overflow="ellipsis", max_width=40)
            for row in rows[:_RESULT_PREVIEW_LINES]:
                table.add_row(*row)
            body.append(table)
            if len(rows) > _RESULT_PREVIEW_LINES:
                body.append(f"\n  ... {len(rows) - _RESULT_PREVIEW_LINES} more rows", style="dim")
        else:
            body.append(f"  {_first_line(content)[:300]}", style="dim")
        return body

    def _format_graph_result(self, content: str) -> Text:
        body = Text()
        m = re.search(r"\*\*(\d+) row\(s\)\*\*", content)
        if m:
            body.append(f"  {m.group(1)} rows", style="dim")

        lines = content.strip().split("\n")
        data_rows: list[str] = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            if stripped.startswith("|") and "---" not in stripped:
                data_rows.append(stripped)

        if len(data_rows) > 1:
            if body:
                body.append("\n")
            for row in data_rows[1:_RESULT_PREVIEW_LINES + 1]:
                body.append(f"  {row[:200]}\n", style="dim")
            if len(data_rows) - 1 > _RESULT_PREVIEW_LINES:
                body.append(f"  ... {len(data_rows) - 1 - _RESULT_PREVIEW_LINES} more rows", style="dim")
        elif not m:
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("*") and not stripped.startswith("```"):
                    prefix = "red" if stripped.startswith("Query failed") else "dim"
                    body.append(f"  {stripped[:300]}", style=prefix)
                    break
        return body if body else None

    # ------------------------------------------------------------------ #
    #  helpers
    # ------------------------------------------------------------------ #

    def _flush_think(self) -> None:
        if self._think_open:
            self._think_open = False
            self._console.print()


def _first_line(content: str) -> str:
    for line in content.strip().split("\n"):
        s = line.strip()
        if s:
            return s
    return ""
