# app/cli/commands/init.py
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from app.cli.i18n import I18nLoader


async def run_init(model_path: str, lang: str, console: Console) -> None:
    from app.db.base import create_sql_executor
    from app.semantics.graph import init_all_graphs
    from app.semantics.graph.entity.flow import CliProgressConsumer
    from app.semantics.models import OSISpecification

    i18n = I18nLoader(lang)
    console.print(f"\n[bold]{i18n.t('cli.init.starting')}[/]\n")

    osi = OSISpecification.load_from_yaml(Path(model_path))
    model_source = osi.semantic_model[0]

    executor = create_sql_executor("default")

    console.print(f"  {i18n.t('cli.init.sampling')}")

    await init_all_graphs(
        model_source,
        executor,
        entity_consumers=[CliProgressConsumer()],
    )

    console.print(f"\n[bold green]{i18n.t('cli.init.all_ready')}[/]")
    Prompt.ask(f"\n{i18n.t('cli.init.press_enter')}", default="")
