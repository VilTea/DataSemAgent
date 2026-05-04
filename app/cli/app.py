# app/cli/app.py
import asyncio
from pathlib import Path

import typer

app = typer.Typer(help="DataSemAgent CLI", invoke_without_command=True)


def _resolve_model_path(cli_arg: str | None) -> str:
    if cli_arg:
        return cli_arg
    from app.config import config
    path = config.main_config.paths.get("semantics")
    if not path:
        raise SystemExit("Specify model path via --model or config.toml paths.semantics")
    return str(path)


def _run_menu_loop(model_path: str, lang: str) -> None:
    from rich.console import Console
    from rich.prompt import Prompt

    from app.cli.i18n import I18nLoader

    i18n = I18nLoader(lang)
    console = Console()

    while True:
        console.clear()
        console.print("[bold cyan]DataSemAgent[/]\n")
        console.print(f"  [1] [bold]init[/]  {i18n.t('cli.menu.init')}")
        console.print(f"  [2] [bold]ask[/]  {i18n.t('cli.menu.ask')}")
        console.print(f"  [3] [bold]exit[/]  {i18n.t('cli.menu.exit')}\n")

        choice = Prompt.ask(
            i18n.t("cli.menu.choose"),
            choices=["1", "2", "3"],
            default="2",
        )

        if choice == "1":
            from app.cli.commands.init import run_init
            asyncio.run(run_init(model_path, lang, console))
        elif choice == "2":
            from app.cli.commands.ask import run_ask
            asyncio.run(run_ask(model_path, lang, console))
        elif choice == "3":
            console.print(f"\n{i18n.t('cli.menu.goodbye')}")
            break


@app.callback()
def callback(
    lang: str = typer.Option("zh", "--lang", "-l", help="UI language (zh/en)"),
    model: str = typer.Option(None, "--model", "-m", help="Path to OSI model yaml"),
):
    """Launch the DataSemAgent interactive terminal."""
    model_path = _resolve_model_path(model)
    if not Path(model_path).exists():
        from app.cli.i18n import I18nLoader
        i18n = I18nLoader(lang)
        raise SystemExit(i18n.t("cli.error.model_not_found", path=model_path))

    _detect_graphs_on_startup()
    _run_menu_loop(model_path, lang)


def _detect_graphs_on_startup() -> None:
    """Check if graphs already exist on disk and update init_state.

    Does NOT trigger any graph building — pure detection.
    Uses executor factories, no hardcoded implementation details.
    """
    from app.logger import logger
    from app.semantics.graph.init import _METRIC_GRAPH_PATH
    from app.semantics.graph.init_state import init_state
    from app.semantics.graph.loader import KuzuLoader, create_graph_loader

    # Metric graph
    if _executor_has_data(KuzuLoader(path=_METRIC_GRAPH_PATH)):
        loader = KuzuLoader(path=_METRIC_GRAPH_PATH)
        init_state.set_executor("metric_graph", loader)
        init_state.mark_ready("metric_graph")
        logger.info("Metric graph detected on startup")

    # Entity graph (uses config-driven factory)
    loader = create_graph_loader("default")
    if _executor_has_data(loader):
        init_state.set_executor("entity_graph", loader)
        init_state.mark_ready("entity_graph")
        # Restore persisted schema so entity descriptions survive restarts
        from app.semantics.graph.entity.nodes import load_entity_schema
        schema = load_entity_schema(loader)
        if schema is not None:
            init_state.set_extra("entity_graph", "schema", schema)
        logger.info("Entity graph detected on startup")


def _executor_has_data(executor) -> bool:
    """True if *executor* already contains node tables."""
    try:
        result = executor.execute("CALL show_tables() RETURN *")
        while result.has_next():
            if result.get_next()[2] == "NODE":
                return True
        return False
    except Exception:
        return False


def main() -> None:
    app()
