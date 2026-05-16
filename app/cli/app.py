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
        console.print(f"  [8] [bold]inspect[/]  {i18n.t('cli.menu.inspect')}")
        console.print(f"  [9] [bold]build-db[/]  {i18n.t('cli.menu.build_db')}\n")

        choice = Prompt.ask(
            i18n.t("cli.menu.choose"),
            choices=["1", "2", "3", "8", "9"],
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
        elif choice == "8":
            _run_inspect(lang, console)
        elif choice == "9":
            _run_build_db(lang, console)


@app.callback()
def callback(
    lang: str = typer.Option("zh", "--lang", "-l", help="UI language (zh/en)"),
    model: str = typer.Option(None, "--model", "-m", help="Path to OSI model yaml"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable verbose hook logging"),
):
    """Launch the DataSemAgent interactive terminal."""
    if debug:
        from app.hook.registry import HookRegistry
        HookRegistry.set_debug(True)
        from app.logger import logger
        logger.info("Debug mode enabled — hook triggers will be logged")

    model_path = _resolve_model_path(model)
    if not Path(model_path).exists():
        from app.cli.i18n import I18nLoader
        i18n = I18nLoader(lang)
        raise SystemExit(i18n.t("cli.error.model_not_found", path=model_path))

    _detect_graphs_on_startup()
    _run_menu_loop(model_path, lang)


def _detect_graphs_on_startup() -> None:
    """No-op — graph readiness is determined lazily by each flow on first use.

    KuzuDB uses file-level locks so we must NOT open connections here.
    Filesystem checks would couple to implementation details (file vs dir).
    Both approaches violate the GraphExecutor abstraction.

    Each flow (init / ask / inspect) opens the connection it needs via
    create_graph_loader() and registers it in init_state on success.
    """
    pass


def _run_inspect(lang: str, console) -> None:
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text
    from app.cli.i18n import I18nLoader
    from app.semantics.graph.init_state import init_state
    from app.semantics.graph.loader import create_graph_loader

    i18n = I18nLoader(lang)

    # Lazy-open: startup doesn't pre-open connections.
    if not init_state.is_ready("reasoning_graph"):
        from app.semantics.graph.init import _executor_has_data
        loader = create_graph_loader("reasoning")
        if _executor_has_data(loader):
            init_state.set_executor("reasoning_graph", loader)
            init_state.mark_ready("reasoning_graph")
        else:
            loader.close()
            console.print(f"\n[bold]{i18n.t('cli.inspect.title')}[/]\n")
            console.print(f"  [dim]{i18n.t('cli.inspect.not_ready')}[/]")
            Prompt.ask(f"\n{i18n.t('cli.init.press_enter')}", default="")
            return

    loader = init_state.get_executor("reasoning_graph")

    # Load all facts with full properties once.
    all_facts: list[dict] = []
    try:
        r = loader.execute(
            "MATCH (f:Fact) RETURN f.id, f.content, f.confidence, f.is_ontology, "
            "f.parent_id, f.merged_into, f.reflection_notes ORDER BY f.confidence DESC"
        )
        while r.has_next():
            row = r.get_next()
            all_facts.append({
                "id": str(row[0]), "content": str(row[1]),
                "confidence": row[2] if row[2] is not None else 0.0,
                "is_ontology": bool(row[3]) if len(row) > 3 else False,
                "parent_id": str(row[4]) if len(row) > 4 and row[4] else "",
                "merged_into": str(row[5]) if len(row) > 5 and row[5] else "",
                "reflection_notes": str(row[6]) if len(row) > 6 and row[6] else "",
            })
    except Exception:
        # Fallback: old schema — load whatever properties exist
        try:
            r = loader.execute("MATCH (f:Fact) RETURN f.id, f.content ORDER BY f.id")
            while r.has_next():
                row = r.get_next()
                all_facts.append({
                    "id": str(row[0]), "content": str(row[1]),
                    "confidence": 0.0, "is_ontology": False,
                    "parent_id": "", "merged_into": "", "reflection_notes": "",
                })
        except Exception:
            pass

    if not all_facts:
        console.print(f"\n[bold]{i18n.t('cli.inspect.title')}[/]\n")
        console.print(f"  [dim]{i18n.t('cli.inspect.not_ready')}[/]")
        Prompt.ask(f"\n{i18n.t('cli.init.press_enter')}", default="")
        return

    PAGE_SIZE = 20

    def _show_overview(page: int = 0):
        console.clear()
        console.print(f"\n[bold]{i18n.t('cli.inspect.title')}[/]\n")
        total = len(all_facts)
        total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        start = page * PAGE_SIZE
        page_facts = all_facts[start:start + PAGE_SIZE]

        body = Text()
        for i, f in enumerate(page_facts, start + 1):
            body.append(f"{i:>2}. ", style="dim")
            if f["is_ontology"]:
                body.append("[ONT] ", style="bold cyan")
            body.append(f"{f['content'][:100]}\n")
        title = f"Facts ({start + 1}-{start + len(page_facts)} / {total})"
        console.print(Panel(body, title=title, border_style="dim", padding=(0, 1)))

        if total_pages > 1:
            console.print(f"  [dim]p{page + 1}/{total_pages}  n=下一页  p=上一页[/]")

        # Edge counts
        edge_parts = []
        for label in ("input_to", "outputs", "references", "sourced_from", "equivalent_to"):
            try:
                r2 = loader.execute(f"MATCH ()-[e:{label}]->() RETURN count(*) AS cnt")
                while r2.has_next():
                    cnt = r2.get_next()[0]
                    if cnt > 0:
                        edge_parts.append(f"{label}: {cnt}")
            except Exception:
                pass
        if edge_parts:
            console.print(f"  [dim]{', '.join(edge_parts)}[/]")

        return total_pages

    def _show_detail(fact: dict):
        console.clear()
        fid = fact["id"]
        console.print(f"\n[bold]{i18n.t('cli.inspect.fact_detail')}[/]\n")

        # Basic info
        console.print(f"  [bold]{i18n.t('cli.inspect.content')}:[/] {fact['content']}")
        console.print(f"  [bold]{i18n.t('cli.inspect.confidence')}:[/] {fact['confidence']:.0%}")
        if fact["is_ontology"]:
            console.print(f"  [bold cyan]{i18n.t('cli.inspect.ontology')}[/]")

        # Resolve parent/merged_into to human-readable content
        if fact["parent_id"]:
            parent_content = _lookup_content(fact["parent_id"])
            console.print(f"  [bold]{i18n.t('cli.inspect.parent')}:[/] {parent_content}")
        if fact["merged_into"]:
            merged_content = _lookup_content(fact["merged_into"])
            console.print(f"  [bold]{i18n.t('cli.inspect.merged_into')}:[/] {merged_content}")
        if fact["reflection_notes"]:
            console.print(f"  [dim]{fact['reflection_notes']}[/]")

        # Query all edges connected to this fact.
        # Edge properties (dependency, merged) only exist on specific edge types;
        # query them separately to avoid binder errors on types that lack them.
        edges_out: list[dict] = []
        edges_in: list[dict] = []
        for label in ("input_to", "outputs", "references", "sourced_from", "equivalent_to"):
            try:
                r = loader.execute(
                    f"MATCH (a:Fact {{id: '{_esc(fid)}'}})-[e:{label}]->(b) RETURN b.id"
                )
                while r.has_next():
                    target_id = str(r.get_next()[0])
                    target_text = _lookup_content(target_id)
                    detail = _edge_detail(label, fid, target_id, outgoing=True)
                    edges_out.append({"label": label, "dir": "→", "target": target_text, "detail": detail})
            except Exception:
                pass
            try:
                r = loader.execute(
                    f"MATCH (a)-[e:{label}]->(b:Fact {{id: '{_esc(fid)}'}}) RETURN a.id"
                )
                while r.has_next():
                    source_id = str(r.get_next()[0])
                    source_text = _lookup_content(source_id)
                    detail = _edge_detail(label, source_id, fid, outgoing=False)
                    edges_in.append({"label": label, "dir": "←", "source": source_text, "detail": detail})
            except Exception:
                pass

        if edges_out or edges_in:
            console.print(f"\n  [bold]Edges:[/]")
            for e in edges_out:
                d = f" [dim]({e['detail']})[/]" if e["detail"] else ""
                console.print(f"    → [{e['label']}]{d} {e['target'][:100]}")
            for e in edges_in:
                d = f" [dim]({e['detail']})[/]" if e["detail"] else ""
                console.print(f"    {e['source'][:100]} [{e['label']}]{d} → (this)")

        if not edges_out and not edges_in:
            console.print(f"\n  [dim]{i18n.t('cli.inspect.no_details')}[/]")

    def _lookup_content(node_id: str) -> str:
        """Look up human-readable content for a node by id (probes each label)."""
        # Each label has different properties — query only what exists on that label.
        probes = [
            ("Fact", "content"),
            ("ReasoningStep", "description"),
            ("OSIRef", "osi_name"),
            ("Source", "query_sql"),
        ]
        for label, prop in probes:
            try:
                r = loader.execute(
                    f"MATCH (n:{label}) WHERE n.id = '{_esc(node_id)}' "
                    f"RETURN n.{prop} LIMIT 1"
                )
                while r.has_next():
                    val = r.get_next()[0]
                    if val:
                        return str(val)
            except Exception:
                continue
        return f"<{node_id}>"

    def _edge_detail(label: str, from_id: str, to_id: str, outgoing: bool) -> str:
        """Get edge property details (dependency / merged) without cross-type errors."""
        try:
            if label == "input_to":
                if outgoing:
                    r = loader.execute(
                        f"MATCH (a:Fact {{id: '{_esc(from_id)}'}})-[e:input_to]->(b {{id: '{_esc(to_id)}'}}) "
                        f"RETURN e.dependency"
                    )
                else:
                    r = loader.execute(
                        f"MATCH (a {{id: '{_esc(from_id)}'}})-[e:input_to]->(b:Fact {{id: '{_esc(to_id)}'}}) "
                        f"RETURN e.dependency"
                    )
                while r.has_next():
                    dep = r.get_next()[0]
                    if dep:
                        return f"dependency={dep}"
            elif label == "equivalent_to":
                if outgoing:
                    r = loader.execute(
                        f"MATCH (a:Fact {{id: '{_esc(from_id)}'}})-[e:equivalent_to]->(b {{id: '{_esc(to_id)}'}}) "
                        f"RETURN e.merged"
                    )
                else:
                    r = loader.execute(
                        f"MATCH (a {{id: '{_esc(from_id)}'}})-[e:equivalent_to]->(b:Fact {{id: '{_esc(to_id)}'}}) "
                        f"RETURN e.merged"
                    )
                while r.has_next():
                    if r.get_next()[0]:
                        return "merged"
        except Exception:
            pass
        return ""

    def _esc(val: str) -> str:
        return val.replace("\\", "\\\\").replace("'", "\\'")

    # ── navigation ──
    page = 0
    total_pages = _show_overview(page)
    while True:
        choice = Prompt.ask(f"\n{i18n.t('cli.inspect.enter_number')}", default="").strip().lower()

        if choice in ("/exit",):
            break

        # Overview mode: Enter or /back exits to menu
        if choice in ("", "/back"):
            break

        # Pagination (only in overview mode)
        if choice == "n":
            page = min(page + 1, total_pages - 1)
            total_pages = _show_overview(page)
            continue
        elif choice == "p":
            page = max(page - 1, 0)
            total_pages = _show_overview(page)
            continue

        # Number selection
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(all_facts):
                _show_detail(all_facts[idx])
                # Detail mode: Enter → back to overview, number → jump to that fact
                while True:
                    inner = Prompt.ask(
                        f"\n[{idx + 1}] {all_facts[idx]['content'][:60]}...  [dim]序号跳转 | Enter 返回列表[/]",
                        default=""
                    ).strip().lower()
                    if inner in ("", "/back"):
                        break
                    if inner in ("/exit",):
                        return
                    try:
                        idx = int(inner) - 1
                        if 0 <= idx < len(all_facts):
                            _show_detail(all_facts[idx])
                        else:
                            console.print(f"  [yellow]1-{len(all_facts)}[/]")
                    except ValueError:
                        console.print(f"  [yellow]1-{len(all_facts)}[/]")
                total_pages = _show_overview(page)
            else:
                console.print(f"  [yellow]1-{len(all_facts)}[/]")
        except ValueError:
            console.print(f"  [yellow]1-{len(all_facts)} n/p 翻页 /back 返回[/]")


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
