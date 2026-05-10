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
        console.print(f"  [3] [bold]exit[/]  {i18n.t('cli.menu.exit')}")
        console.print(f"  [9] [bold]build-db[/]  {i18n.t('cli.menu.build_db')}\n")

        choice = Prompt.ask(
            i18n.t("cli.menu.choose"),
            choices=["1", "2", "3", "9"],
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
        elif choice == "9":
            _run_build_db(lang, console)


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


def _run_build_db(lang: str, console) -> None:
    from pathlib import Path
    from rich.prompt import Confirm, Prompt
    from app.cli.i18n import I18nLoader
    from app.config import PROJECT_ROOT, config

    i18n = I18nLoader(lang)

    # Build script reads from the OSI model; compare paths.
    build_model = str(PROJECT_ROOT / "config" / "semantics" / "tpcds_model_sqlite.yaml")
    build_script = str(PROJECT_ROOT / "tests" / "build_tpcds_test_data.py")
    build_db_path = str(PROJECT_ROOT / "data" / "test.db")

    cfg_model = config.main_config.paths.get("semantics", "")
    db_cfg = config.database.get("default")
    cfg_db_path = str(db_cfg.specific.path) if db_cfg else ""
    cfg_db_type = getattr(db_cfg, "type", None) if db_cfg else None

    console.print(f"\n[bold]{i18n.t('cli.build_db.starting')}[/]\n")

    # Model path comparison
    model_match = str(PROJECT_ROOT / cfg_model) == build_model if cfg_model else False
    if model_match:
        console.print(f"  [green] {i18n.t('cli.build_db.model')} {cfg_model} (同)[/]")
    else:
        console.print(f"  [yellow] {i18n.t('cli.build_db.model')}[/]")
        console.print(f"    {i18n.t('cli.build_db.current')}: {cfg_model}")
        console.print(f"    {i18n.t('cli.build_db.build')}: {build_model}")

    # DB path comparison
    if cfg_db_type and cfg_db_type != "sqlite":
        console.print(f"  [yellow] {i18n.t('cli.build_db.db_type', type=cfg_db_type)}[/]")
    elif str(PROJECT_ROOT / cfg_db_path) == build_db_path if cfg_db_path else False:
        console.print(f"  [green] {i18n.t('cli.build_db.db')} {cfg_db_path} (同)[/]")
    else:
        console.print(f"  [yellow] {i18n.t('cli.build_db.db')}[/]")
        console.print(f"    {i18n.t('cli.build_db.current')}: {cfg_db_path}")
        console.print(f"    {i18n.t('cli.build_db.build')}: {build_db_path}")

    console.print()
    if not Confirm.ask(i18n.t("cli.build_db.confirm"), default=True):
        return

    console.print(f"\n  {i18n.t('cli.build_db.running')}")

    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(build_script), "--db", str(build_db_path)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        console.print(result.stdout)
        console.print(f"\n[bold green]{i18n.t('cli.build_db.done')}[/]")
    else:
        console.print(f"\n[red]{result.stderr}[/]")

    Prompt.ask(f"\n{i18n.t('cli.init.press_enter')}", default="")


def main() -> None:
    app()
