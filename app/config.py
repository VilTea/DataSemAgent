import json
import sys
import threading
import tomllib
from enum import Enum
from pathlib import Path
from typing import ClassVar, Any, TypeVar, Union, Protocol, runtime_checkable, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from app.exceptions import DataSemAgentError


@runtime_checkable
class ConfigBase(Protocol):
    setting_name: ClassVar[str]
    base_path: ClassVar[Path]
    file_type: ClassVar[tuple[str]]
    load_priority: ClassVar[int]

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = get_project_root()

_CONFIG_REGISTRY: list[type] = []

def register_config(cls: type) -> type:
    """Decorator: Register configuration class"""
    if isinstance(cls, ConfigBase):
        _CONFIG_REGISTRY.append(cls)
    return cls

def get_registered_configs() -> list[type]:
    """Get all registered configuration classes"""
    return _CONFIG_REGISTRY.copy()

@register_config
class MainSetting(BaseModel):
    paths: dict[str, str | Path] = Field(default={})

    setting_name: ClassVar[str] = "main"
    base_path: ClassVar[Path] = PROJECT_ROOT / "config" / "config.toml"
    file_type: ClassVar[tuple[str]] = ("toml",)
    load_priority: ClassVar[int] = sys.maxsize

    @model_validator(mode="after")
    def initialize(self) -> "MainSetting":
        self.paths = {
            k: Path(v) if v.startswith(str(PROJECT_ROOT)) else PROJECT_ROOT / v
            for k, v in self.paths.items()
        }
        return self

@register_config
class LLMSettings(BaseModel):
    type: str = Field(..., description="Model type. Such as openai, anthropic...")
    model: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(default=4096, description="Maximum number of tokens per request")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    api_type: str = Field(default="openai", description="Azure, Openai, or Ollama")
    api_version: str | None = Field(default=None, description="Azure Openai version if AzureOpenai")
    args: dict[str, Any] = Field(default={}, description="Arguments")
    context_window: int = Field(
        default=0, ge=0,
        description="Model context window in tokens. 0 disables compression.",
    )

    setting_name: ClassVar[str] = "llm"
    base_path: ClassVar[Path] = PROJECT_ROOT / "config" / "llm" / "config.toml"
    file_type: ClassVar[tuple[str]] = ("toml", )
    load_priority: ClassVar[int] = 100

class MCPServerToolSettings(BaseModel):
    name: str = Field(..., description="Tool name")
    description: str = Field(default="", description="Tool description")
    parameters: dict = Field(default_factory=dict, description="Tool parameter schema")

@register_config
class MCPSettings(BaseModel):
    transport_type: str = Field(default="streamable-http", description="Transport type")
    url: str | None = Field(default=None, description="Server URL")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP request headers")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    command: str | None = Field(default=None, description="stdio command")
    args: list[str] = Field(default_factory=list, description="Command line arguments")
    env: dict[str, str] | None = Field(default=None, description="Environment variables")
    cwd: str | None = Field(default=None, description="Working directory")
    module_path: str | None = Field(default=None, description="Module path")
    auto_discover: bool = Field(default=False, description="Whether to auto-discover tools")
    tools: list[MCPServerToolSettings] = Field(default_factory=list, description="Manually configured tool list")

    setting_name: ClassVar[str] = "mcp"
    base_path: ClassVar[Path] = PROJECT_ROOT / "config" / "mcp" / "servers.yaml"
    file_type: ClassVar[tuple[str]] = ("yaml", "toml", "json")
    load_priority: ClassVar[int] = 50

    def model_post_init(self, __context) -> None:
        tp = self.transport_type
        if tp in ["streamable-http", "sse"] and not self.url:
            raise ValueError(f"{tp} requires url")
        if tp == "stdio" and not self.command:
            raise ValueError("stdio requires command")
        if tp in ["python-module", "node-module"] and not self.module_path:
            raise ValueError(f"{tp} requires module_path")

@register_config
class AgentSettings(BaseModel):
    reflection_interval: int = Field(default=10, ge=0, description="Trigger reflection every N turns. 0 disables reflection.")
    compression_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Trigger ratio of LLMSettings.context_window. 0 disables.",
    )
    compression_keep_recent_turns: int = Field(
        default=3, ge=1,
        description="Turns to keep unmodified at the end of conversation.",
    )
    compression_summary_prompt: str = Field(
        default="",
        description="Custom prompt for summarization; empty = built-in default.",
    )

    setting_name: ClassVar[str] = "agent"
    base_path: ClassVar[Path] = PROJECT_ROOT / "config" / "agent.toml"
    file_type: ClassVar[tuple[str]] = ("toml",)
    load_priority: ClassVar[int] = 30


@register_config
class EvalSettings(BaseModel):
    enabled: bool = Field(default=True, description="Enable evaluation data collection")
    output_dir: str = Field(default="data/eval", description="JSONL output directory")
    redact_keys: list[str] = Field(
        default_factory=lambda: ["api_key", "password", "token", "secret", "authorization"],
        description="Keys to redact from collected data",
    )
    token_counting: str = Field(
        default="hybrid",
        description="Token counting strategy: 'api' | 'tiktoken' | 'hybrid'",
    )

    setting_name: ClassVar[str] = "eval"
    base_path: ClassVar[Path] = PROJECT_ROOT / "config" / "eval.toml"
    file_type: ClassVar[tuple[str]] = ("toml",)
    load_priority: ClassVar[int] = 10


""" Database """

class GraphDatabaseType(str, Enum):
    KUZU = "kuzu"


class KuzuSpecificSettings(BaseModel):
    path: str | Path = Field(default="data/entity_graph")


@register_config
class GraphDatabaseSettings(BaseModel):
    setting_name: ClassVar[str] = "graph_database"
    base_path: ClassVar[Path] = PROJECT_ROOT / "config" / "graph_database.toml"
    file_type: ClassVar[tuple[str]] = ("toml",)
    load_priority: ClassVar[int] = 20

    type: GraphDatabaseType = Field(default=GraphDatabaseType.KUZU, description="Graph database type")
    driver: Literal["kuzu"] = Field(default="kuzu")
    specific: KuzuSpecificSettings = Field(default_factory=KuzuSpecificSettings)

    @model_validator(mode="after")
    def normalize_paths(self) -> "GraphDatabaseSettings":
        path = Path(self.specific.path)
        if not path.is_absolute():
            self.specific.path = PROJECT_ROOT / path
        return self


class DatabaseType(str, Enum):
    SQLITE = "sqlite"


class SQLiteSpecificSettings(BaseModel):
    """SQLite-specific configuration"""
    path: str | Path = Field(default="data/app.db", description="Database file path")
    pool_size: int = Field(default=5, ge=1, le=100, description="Connection pool size")
    timeout: float = Field(default=30.0, description="Connection timeout in seconds")
    check_same_thread: bool = Field(default=False, description="Whether to check same thread")

    # SQLite-specific pragma settings
    pragmas: dict[str, str | int] = Field(
        default_factory=lambda: {
            "journal_mode": "WAL",
            "foreign_keys": "ON",
            "cache_size": -16000,
            "synchronous": "NORMAL",
        },
        description="SQLite pragma"
    )

class PoolSettings(BaseModel):
    min_size: int = Field(default=2, ge=1, description="Minimum connection count")
    max_size: int = Field(default=10, ge=1, description="Maximum connection count")
    max_idle_time: float = Field(default=300.0, description="Maximum idle time in seconds")
    max_lifetime: float = Field(default=3600.0, description="Maximum connection lifetime in seconds")

@register_config
class DatabaseSettings(BaseModel):
    # Basic configuration
    type: DatabaseType = Field(default=DatabaseType.SQLITE, description="Database type")
    driver: Literal["aiosqlite", "asyncpg", "pysqlite"] = Field(default="aiosqlite")
    echo: bool = Field(default=False, description="Whether to print SQL statements")
    echo_pool: bool = Field(default=False, description="Whether to print connection pool logs")

    # specific configuration
    specific: SQLiteSpecificSettings | None = Field(default_factory=SQLiteSpecificSettings)

    # Common configuration
    host: str | None = Field(default=None, description="Database host address")
    port: int | None = Field(default=None, description="Database port")
    database: str | None = Field(default=None, description="Database name")
    username: str | None = Field(default=None, description="Username")
    password: str | None = Field(default=None, description="Password")

    # Connection pool configuration
    pool: PoolSettings = Field(default_factory=PoolSettings)

    # Extended configuration (free-form fields)
    extra: dict = Field(default_factory=dict, description="Additional configuration items")

    # Required by ConfigBase protocol
    setting_name: ClassVar[str] = "database"
    base_path: ClassVar[Path] = PROJECT_ROOT / "config" / "database.toml"
    file_type: ClassVar[tuple[str]] = ("toml",)
    load_priority: ClassVar[int] = 80

    @model_validator(mode="after")
    def normalize_paths(self) -> "DatabaseSettings":
        if self.type == DatabaseType.SQLITE:
            path = Path(self.specific.path)
            if not path.is_absolute():
                self.specific.path = PROJECT_ROOT / path
        return self

    def get_connection_url(self) -> str:
        """
        Generate database connection URL
        """
        if self.type == DatabaseType.SQLITE:
            return f"sqlite+{self.driver}:///{str(self.specific.path).replace('\\', '/')}"
        # TODO: Extend support for other database types here
        return f"{self.type}+{self.driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

""" ConfigLoader """

# Define generic type for ConfigBase implementations. Used only for type hints.
T = TypeVar("T", bound=Union[MainSetting, LLMSettings, MCPSettings, DatabaseSettings])

class ConfigLoader:
    """
    Configuration loader: Reads configuration files for all classes implementing ConfigBase protocol.
    """

    def __init__(self, config_class: type[T], main_config: MainSetting = None):
        """
        Initialize configuration loader.

        Args:
            config_class: Configuration class implementing ConfigBase protocol.
        """
        if not isinstance(config_class, ConfigBase):
            raise RuntimeError(f"{config_class.__name__} must implement ConfigBase protocol")

        self.config_class = config_class
        main_paths = main_config.paths if main_config else {}
        self.base_path: Path = main_paths.get(config_class.setting_name) or config_class.base_path
        self.file_type: tuple[str] = config_class.file_type

        # Ensure base path exists
        if not self.base_path or not self.base_path.exists():
            raise RuntimeError(f"Config path does not exist: {self.base_path}")

    def load(self) -> dict[str, T]:
        """
        Load all configurations

        Returns:
            Dictionary of configuration instances, keyed by configuration name/identifier
        """
        if self.base_path.is_file():
            return self._load_single_file()
        else:
            return self._load_multiple_files()

    def _load_single_file(self) -> dict[str, T]:
        """
        Single file with multiple instances mode:
        Each file contains multiple configuration instances, filename serves as category/group
        """
        instances: dict[str, T] = {}
        config_data = self.read_config_file(self.base_path)

        if isinstance(config_data, dict):
            for k, v in config_data.items():
                if isinstance(v, dict):
                    instances[k] = self._create_instance(v)
                else:
                    # Single configuration case
                    instances['default'] = self._create_instance(config_data)
                    break
        elif isinstance(config_data, list):
            for idx, item in enumerate(config_data):
                if isinstance(item, dict):
                    instances[str(idx)] = self._create_instance(item)
        else:
            raise RuntimeError(f"Unexpected config format in {self.base_path}")

        return instances

    def _load_multiple_files(self) -> dict[str, T]:
        """
        Single directory with multiple instances mode:
        Each file corresponds to one configuration instance, filename serves as instance identifier
        """
        instances: dict[str, T] = {}
        for file_path in [file_path for file_type in self.file_type for file_path in self.base_path.glob(f"*.{file_type}")]:
            try:
                config_data = self.read_config_file(file_path)
                if isinstance(config_data, dict):
                    for k, v in config_data.items():
                        if isinstance(v, dict):
                            instances[k] = self._create_instance(v)
                        else:
                            # Single configuration case
                            instances['default'] = self._create_instance(config_data)
                            break
                elif isinstance(config_data, list):
                    for idx, item in enumerate(config_data):
                        if isinstance(item, dict):
                            instances[str(idx)] = self._create_instance(item)
            except Exception as e:
                raise RuntimeError(f"Failed to load config from {file_path}", e)
        return instances

    def _create_instance(self, data: dict[str, Any]) -> T:
        """Create configuration instance"""
        try:
            return self.config_class(**data)
        except Exception as e:
            raise RuntimeError(f"Failed to create {self.config_class.__name__} instance", e)

    @staticmethod
    def read_config_file(file_path: Path) -> dict | list:
        suffix = file_path.suffix.lower()

        try:
            with open(file_path, 'rb') as f:
                if suffix == '.toml':
                    return tomllib.load(f)
                elif suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif suffix == '.json':
                    return json.load(f)
                else:
                    raise RuntimeError(f"Unsupported file type: {suffix}")
        except Exception as e:
            raise RuntimeError(f"Error reading {file_path}", e)

class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.main_config = None
                    self._configs = {}
                    self._load_initial_config()
                    self._initialized = True

    def __getattr__(self, name: str) -> dict[str, T]:
        if name in self._configs:
            return self._configs[name]
        raise AttributeError(f"Config has no attribute '{name}'")

    def __dir__(self):
        """Support autocompletion"""
        return sorted(list(super().__dir__()) + list(self._configs.keys()))

    def _load_initial_config(self):
        config_types = get_registered_configs()
        sorted_types = sorted(config_types, key=lambda x: x.load_priority, reverse=True)
        for config_type in sorted_types:
            if config_type.setting_name == "main":
                self.main_config = MainSetting(**ConfigLoader.read_config_file(MainSetting.base_path))
            else:
                try:
                    self._configs[config_type.setting_name] = ConfigLoader(config_type, self.main_config).load()
                except Exception as e:
                    raise DataSemAgentError(f"load config type {config_type.__name__}", e)


    def load_config(self, config_type: type[ConfigBase]):
        self._configs[config_type.setting_name] = ConfigLoader(config_type).load()

config = Config()
