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
    from app.tool.skill import create_skill_tool
    from app.tool.sql_exec import SqlExecTool
    from app.cli.console import RichConsumer

    i18n = I18nLoader(lang)

    from app.semantics.graph.init_state import init_state

    osi = OSISpecification.load_from_yaml(Path(model_path))
    model_source = osi.semantic_model[0]

    if not init_state.is_ready("entity_graph") or not init_state.is_ready("metric_graph"):
        console.print(f"\n[yellow]{i18n.t('cli.ask.not_initialized')}[/]")
        Prompt.ask(f"\n{i18n.t('cli.init.press_enter')}", default="")
        return

    tools = [
        SqlExecTool(model_source=model_source),
        MetricGraphTool(),
        EntityGraphTool(),
    ]
    skill_tool = create_skill_tool()
    if skill_tool:
        tools.append(skill_tool)

    pipeline = QueuePipeline()
    pipeline.register(RichConsumer(console))

    agent = AgentNode(
        name="cli_agent",
        system_prompt=REACT_SYSTEM_PROMPT,
        tools=tools,
    )
    flow = react_flow(agent_node=agent, pipeline=pipeline)

    console.print(f"\n[bold cyan]{i18n.t('cli.ask.welcome')}[/]\n")

    while True:
        console.print()  # blank line before prompt
        user_input = Prompt.ask(f"[bold green]{i18n.t('cli.ask.prompt')}[/]")
        if user_input.strip() in ("/exit", "/q", "/quit"):
            break
        if not user_input.strip():
            continue

        try:
            await flow.ask(user_input)
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
