"""Graph loader — re-exports from app.db.graph for backward compatibility."""
from app.db.graph import GraphExecutor as GraphLoader, KuzuExecutor as KuzuLoader, create_graph_executor as create_graph_loader

__all__ = ["GraphLoader", "KuzuLoader", "create_graph_loader"]
