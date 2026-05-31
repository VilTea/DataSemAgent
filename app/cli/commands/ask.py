# app/cli/commands/ask.py
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from app.cli.i18n import I18nLoader

async def run_ask(model_path: str, lang: str, console: Console) -> None:
    from app.flow import react_flow
    from app.node.agent import AgentNode
    from app.pipeline import QueuePipeline
    from app.prompt.react_agent import REACT_SYSTEM_PROMPT
    from app.semantics.models import OSISpecification
    from app.tool.entity_graph import EntityGraphTool
    from app.tool.metric_graph import MetricGraphTool
    from app.tool.reasoning_graph import ReasoningGraphTool
    from app.tool.skill import create_skill_tool
    from app.tool.sql_exec import SqlExecTool
    from app.cli.console import RichConsumer
    from app.eval.collector import EvalCollector

    i18n = I18nLoader(lang)

    from app.semantics.graph.init_state import init_state

    osi = OSISpecification.load_from_yaml(Path(model_path))
    model_source = osi.semantic_model[0]

    # Lazy-open graph connections that startup left untouched (avoids lock races).
    from app.semantics.graph.loader import create_graph_loader
    from app.semantics.graph.init import _executor_has_data

    for key, db_key in [("metric_graph", "metric"), ("entity_graph", "entity"), ("reasoning_graph", "reasoning")]:
        if init_state.is_ready(key):
            continue
        loader = create_graph_loader(db_key)
        if _executor_has_data(loader):
            init_state.set_executor(key, loader)
            init_state.mark_ready(key)
        else:
            loader.close()

    if not init_state.is_ready("entity_graph") or not init_state.is_ready("metric_graph"):
        console.print(f"\n[yellow]{i18n.t('cli.ask.not_initialized')}[/]")
        Prompt.ask(f"\n{i18n.t('cli.init.press_enter')}", default="")
        return

    from app.tool.todo_write import TodoWriteTool

    todo_tool = TodoWriteTool()

    tools = [
        todo_tool,
        SqlExecTool(model_source=model_source),
        MetricGraphTool(),
        EntityGraphTool(),
        ReasoningGraphTool(),
    ]
    skill_tool = create_skill_tool()
    if skill_tool:
        tools.append(skill_tool)

    pipeline = QueuePipeline()
    pipeline.register(RichConsumer(console))
    pipeline.register(EvalCollector())

    from app.pipeline import EventConsumer
    class _TodoPanelConsumer(EventConsumer):
        async def start(self): pass
        async def stop(self): pass
        async def consume(self, event):
            from app.schema import AgentCompletion, Message, Role, FinishReason
            if isinstance(event, AgentCompletion):
                if event.finish_reason and event.finish_reason != FinishReason.NONE:
                    _render_todos(console, todo_tool)
            elif isinstance(event, Message) and event.role == Role.TOOL:
                if getattr(event, 'name', '') == 'todo_write':
                    _render_todos(console, todo_tool)

    pipeline.register(_TodoPanelConsumer())

    agent = AgentNode(
        name="cli_agent",
        system_prompt=REACT_SYSTEM_PROMPT,
        tools=tools,
    )
    flow = react_flow(agent_node=agent, pipeline=pipeline)

    from app.semantics.graph.reasoning.flow import install_reflection_hook
    install_reflection_hook(flow)

    console.print(f"\n[bold cyan]{i18n.t('cli.ask.welcome')}[/]\n")

    while True:
        console.print()  # blank line after conversation
        _render_todos(console, todo_tool)
        console.print("─" * console.width, style="dim")
        user_input = Prompt.ask(f"[bold green]{i18n.t('cli.ask.prompt')}[/]")
        if user_input.strip() in ("/exit", "/q", "/quit"):
            break
        if user_input.strip() == "/reflect":
            await _trigger_reflection(flow)
            continue
        if not user_input.strip():
            continue

        try:
            await flow.ask(user_input)
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/]")
        console.print()  # ensure newline after streamed content
        console.file.flush()


async def _trigger_reflection(flow) -> None:
    from app.semantics.graph.reasoning.flow import _do_reflection
    await _do_reflection(flow.context.memory)


def _render_todos(console, todo_tool) -> None:
    """Render the todo list between conversation and prompt."""
    if todo_tool.is_empty or todo_tool.all_completed:
        return
    from rich.panel import Panel
    from rich.text import Text

    body = Text()
    icon_map = {"completed": ("✓", "green"), "in_progress": ("◉", "yellow"), "pending": ("○", "dim")}
    for item in todo_tool.items:
        icon, icon_style = icon_map.get(item.status, (" ", ""))
        content_style = {"completed": "green", "in_progress": "yellow", "pending": "dim"}.get(item.status, "")
        body.append(f"{icon} ", style=icon_style)
        body.append(f"{item.content}\n", style=content_style)

    console.print(Panel(body, title="Tasks", border_style="dim", padding=(0, 1)))
