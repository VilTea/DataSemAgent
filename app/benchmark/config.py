"""BenchmarkConfig — swaps the global config singleton for isolated benchmark runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any


_BENCHMARK_CONFIG_ROOT = Path(__file__).resolve().parent.parent.parent / "config" / "benchmark"


class BenchmarkConfig:
    """Context manager that overlays benchmark config on the global singleton.

    Usage::

        with BenchmarkConfig():
            from app.config import config
            db = config.database["dabstep"]  # reads config/benchmark/database.toml
    """

    def __init__(self):
        self._saved: dict[str, Any] = {}
        self._active = False

    def __enter__(self) -> "BenchmarkConfig":
        from app.config import config as global_config

        self._saved = dict(global_config._configs)
        self._active = True

        benchmark_main = _BENCHMARK_CONFIG_ROOT / "config.toml"
        if benchmark_main.exists():
            global_config.main_config = _load_main_config(benchmark_main)

        self._load_benchmark_configs(global_config)
        return self

    def __exit__(self, *args) -> None:
        from app.config import config as global_config

        global_config._configs.clear()
        global_config._configs.update(self._saved)
        self._active = False

    @staticmethod
    def _load_benchmark_configs(global_config) -> None:
        from app.config import ConfigLoader, AgentSettings, DatabaseSettings, GraphDatabaseSettings

        for config_class, filename in [
            (DatabaseSettings, "database.toml"),
            (AgentSettings, "agent.toml"),
            (GraphDatabaseSettings, "graph_database.toml"),
        ]:
            path = _BENCHMARK_CONFIG_ROOT / filename
            if not path.exists():
                continue
            try:
                loader = ConfigLoader.__new__(ConfigLoader)
                loader.config_class = config_class
                loader.base_path = path
                loader.file_type = ("toml",)
                global_config._configs[config_class.setting_name] = loader.load()
            except Exception:
                pass


def _load_main_config(path: Path):
    from app.config import MainSetting, ConfigLoader
    data = ConfigLoader.read_config_file(path)
    return MainSetting(**data)
