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

_METRIC_GRAPH_PATH = str(PROJECT_ROOT / "data" / "metric_graph")


async def init_all_graphs(
    model: SemanticModel,
    executor=None,
    graph_db_config_key: str = "default",
    entity_consumers: list[EventConsumer] | None = None,
) -> None:
    """Build all graph pipelines and mark them ready.

    Skips graphs that already exist on disk (from a prior session).

    Args:
        model: The OSI SemanticModel.
        executor: SqlExecutor for reading database tables (required for entity graph).
        graph_db_config_key: Key in config.graph_database for graph DB settings.
        entity_consumers: Progress consumers for the entity graph build.
    """
    await _detect_or_init_metric_graph(model)

    if executor is not None:
        from app.semantics.graph.loader import create_graph_loader
        loader = create_graph_loader(graph_db_config_key)
        await _detect_or_init_entity_graph(model, executor, loader=loader,
                                           consumers=entity_consumers)


# ------------------------------------------------------------------ #
#  detection helpers
# ------------------------------------------------------------------ #

def _graph_has_data(path: str) -> bool:
    """True if *path* is a Kùzu database directory that contains node tables."""
    import kuzu
    p = Path(path).resolve()
    if not p.is_dir() or not any(p.iterdir()):
        return False
    try:
        db = kuzu.Database(str(p))
        conn = kuzu.Connection(db)
        try:
            r = conn.execute("CALL show_tables() RETURN *")
            while r.has_next():
                row = r.get_next()
                if row[2] == "NODE":
                    return True
            return False
        finally:
            conn.close()
            db.close()
    except Exception:
        return False


# ------------------------------------------------------------------ #
#  metric graph
# ------------------------------------------------------------------ #

async def _detect_or_init_metric_graph(model: SemanticModel) -> None:
    from app.semantics.graph.loader import KuzuLoader

    path = _METRIC_GRAPH_PATH
    if _graph_has_data(path):
        logger.info(f"Metric graph found on disk ({path}), reusing")
        loader = KuzuLoader(path=path)
        init_state.set_executor("metric_graph", loader)
        init_state.mark_ready("metric_graph")
        return

    from app.semantics.graph.metric.pipeline import build_metric_graph

    loader = KuzuLoader(path=path)
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
    consumers=None,
) -> None:
    path = loader._path if loader is not None else "data/entity_graph"
    if _graph_has_data(path):
        logger.info(f"Entity graph found on disk ({path}), reusing")
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
