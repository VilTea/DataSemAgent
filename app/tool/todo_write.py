"""TodoWrite tool — full-replacement task list with state machine enforcement."""
from __future__ import annotations

from pydantic import BaseModel, PrivateAttr

from app.hook import HookPoint, hook
from app.schema import ToolCall
from app.tool.base import BaseTool, ToolResult


class TodoItem(BaseModel):
    id: str
    content: str
    status: str = "pending"


class TodoParams(BaseModel):
    todos: list[TodoItem]


class TodoWriteTool(BaseTool):
    permission: str = "agent"
    name: str = "todo_write"
    description: str = (
        "Manage your task list. ALWAYS pass the COMPLETE list — the system "
        "replaces the old list entirely, not merge diffs.\n"
        "WHEN TO CALL: at task start, when you complete a phase, when all "
        "tasks are done, or when plans change.\n"
        "RULES:\n"
        "- Pass the full list every time, not just changes.\n"
        "- Only ONE task in_progress at a time.\n"
        "- Keep task IDs stable across calls.\n"
        "- Content should be concise — one sentence per task.\n"
        "- Every task must end as 'completed'. Don't leave stale tasks.\n"
        "STATUS: pending → in_progress → completed."
    )
    strict: bool = True
    parameters: dict = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Stable task identifier"},
                        "content": {"type": "string", "description": "One-sentence task description"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "Task status",
                        },
                    },
                    "required": ["id", "content", "status"],
                },
            }
        },
        "required": ["todos"],
    }

    _items: list[TodoItem] = PrivateAttr(default_factory=list)
    _turns_since_last_call: int = PrivateAttr(default=0)

    @property
    def items(self) -> list[TodoItem]:
        return self._items

    @property
    def all_completed(self) -> bool:
        return len(self._items) > 0 and all(i.status == "completed" for i in self._items)

    @property
    def is_empty(self) -> bool:
        return len(self._items) == 0

    async def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        try:
            params = TodoParams.model_validate(tool_call.function.arguments_dict)
        except Exception as e:
            return ToolResult.failure_response(tool_call.id, self.name, f"Invalid: {e}")
        has_in_progress = False
        for ti in params.todos:
            if ti.status == "in_progress":
                if has_in_progress:
                    ti.status = "pending"
                has_in_progress = True
            if ti.status not in ("pending", "in_progress", "completed"):
                ti.status = "pending"
        self._items = params.todos
        self._turns_since_last_call = 0
        lines = [f"- [{i.status}] {i.content}" for i in self._items]
        return ToolResult.success_response(
            tool_call.id, self.name,
            f"Todo list updated ({len(self._items)} items):\n" + "\n".join(lines),
        )

    @hook(HookPoint.FLOW_END)
    async def _reminder(self, ctx) -> None:
        """Inject reminder if todo list is non-empty and not called for 5 turns."""
        if self.is_empty:
            return
        self._turns_since_last_call += 1
        if self._turns_since_last_call >= 5:
            from app.schema import Message, Role
            ctx.memory.add_message(Message(
                role=Role.USER,
                content="<reminder>Update your todos via todo_write.</reminder>",
                injected=True,
            ))
