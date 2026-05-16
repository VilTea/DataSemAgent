"""Graph initialization — called at startup to build and register all graphs.

On first run both graphs are built from scratch and persisted to disk.
On subsequent runs the persisted databases are detected and reused,
skipping the expensive rebuild.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.config import PROJECT_ROOT
from app.logger import logger
from app.semantics.graph.init_state import init_state
from app.semantics.models import SemanticModel

if TYPE_CHECKING:
    from app.pipeline.abc import EventConsumer

async def init_all_graphs(
    model: SemanticModel,
    executor=None,
    graph_db_config_key: str = "entity",
    entity_consumers: list[EventConsumer] | None = None,
) -> None:
    """Build all graph pipelines and mark them ready.

    Skips graphs that already exist on disk (from a prior session).
    Only opens each graph database once — no duplicate connections.

    Args:
        model: The OSI SemanticModel.
        executor: SqlExecutor for reading database tables (required for entity graph).
        graph_db_config_key: Key in config.graph_database for graph DB settings.
        entity_consumers: Progress consumers for the entity graph build.
    """
    await _detect_or_init_metric_graph(model)

    if executor is not None:
        await _detect_or_init_entity_graph(model, executor,
                                           graph_db_config_key=graph_db_config_key,
                                           consumers=entity_consumers)


# ------------------------------------------------------------------ #
def _executor_has_data(executor) -> bool:
    """True if *executor* already contains node tables (protocol-compliant)."""
    try:
        result = executor.execute("CALL show_tables() RETURN *")
        while result.has_next():
            if result.get_next()[2] == "NODE":
                return True
        return False
    except Exception:
        return False


# ------------------------------------------------------------------ #
#  metric graph
# ------------------------------------------------------------------ #

async def _detect_or_init_metric_graph(model: SemanticModel) -> None:
    from app.semantics.graph.loader import create_graph_loader

    if init_state.is_ready("metric_graph"):
        return

    loader = create_graph_loader("metric")
    if _executor_has_data(loader):
        logger.info("Metric graph found on disk, reusing")
        init_state.set_executor("metric_graph", loader)
        init_state.mark_ready("metric_graph")
        return

    from app.semantics.graph.metric.pipeline import build_metric_graph

    await build_metric_graph(model, loader=loader)
    init_state.set_executor("metric_graph", loader)
    init_state.mark_ready("metric_graph")


# ------------------------------------------------------------------ #
#  entity graph
# ------------------------------------------------------------------ #

async def _detect_or_init_entity_graph(
    model: SemanticModel,
    executor,
    loader=None,
    graph_db_config_key: str = "entity",
    consumers=None,
) -> None:
    from app.semantics.graph.loader import create_graph_loader

    if init_state.is_ready("entity_graph"):
        return

    if loader is None:
        loader = create_graph_loader(graph_db_config_key)

    if _executor_has_data(loader):
        logger.info("Entity graph found on disk, reusing")
        init_state.set_executor("entity_graph", loader)
        init_state.mark_ready("entity_graph")
        return

    from app.semantics.graph.entity.flow import init_entity_graph

    await init_entity_graph(
        model=model,
        executor=executor,
        loader=loader,
        consumers=consumers,
    )


# ------------------------------------------------------------------ #
#  sync helpers
# ------------------------------------------------------------------ #

def init_metric_graph_sync(model: SemanticModel) -> None:
    import asyncio
    asyncio.run(_detect_or_init_metric_graph(model))
