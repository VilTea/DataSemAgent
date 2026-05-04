import re
from pathlib import Path

import loguru
import yaml
from pydantic import BaseModel, Field, ConfigDict, model_validator

from app.logger import logger


def _safe_read_text(path: Path, _logger: loguru.logger = None) -> str | None:
    """安全读取文本文件，处理编码错误"""
    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        if _logger:
            _logger.error(f"File {path} is not valid UTF-8")
            return None
        else:
            return f"File {path} is not valid UTF-8"
    except Exception as e:
        if _logger:
            _logger.exception(f"Failed to read {path}", e)
            return None
        else:
            return f"Failed to read {path}, error: {e}"


# 路径
from app.config import config
SKILLS_PATH = config.main_config.paths.get("skills")

# 常量定义
SKILL_FILENAME = "SKILL.md"

# 文件扩展名配置
REFERENCE_EXTENSIONS = {'.md', '.txt', '.json', '.yaml', '.yml'}
SCRIPT_EXTENSIONS = {'.py', '.sh', '.js'}

# Regex: lowercase letters, numbers, hyphens. No leading/trailing hyphens. No consecutive hyphens.
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class Skill(BaseModel):
    """Agent Skills spec compliant skill entity.

    See https://agentskills.io/specification
    """

    # Metadata
    id: str = Field(..., description="Unique identifier (name@version)")
    path: Path = Field(..., description="Skill directory path (parent of SKILL.md)")
    instructions: str = Field(..., description="Skill instruction body (SKILL.md body content)")

    # Specification (see https://agentskills.io/specification)
    name: str = Field(
        ...,
        description="Max 64 characters. Lowercase letters, numbers, and hyphens only. "
        "Must not start or end with a hyphen. Must not contain consecutive hyphens. "
        "Must match the parent directory name.",
        max_length=64,
    )
    description: str = Field(
        ...,
        description="Max 1024 characters. Describes what the skill does and when to use it.",
        min_length=1,
        max_length=1024,
    )
    # Optional
    license: str | None = Field(default=None)
    compatibility: str | None = Field(default=None, max_length=500)
    metadata: dict[str, str] | None = Field(default=None)
    allowed_tools: list[str] | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def display_name(self) -> str:
        return self.name.replace("-", " ").title()

    @property
    def location(self) -> Path:
        """Absolute path to the SKILL.md file (agentskills.io spec: 'location')."""
        return self.path / SKILL_FILENAME

    @property
    def catalog_entry(self) -> dict:
        """Tier 1 progressive disclosure: name + description (~50-100 tokens)."""
        return {"name": self.name, "description": self.description}

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #

    @model_validator(mode="after")
    def _validate_name(self) -> "Skill":
        if not _SKILL_NAME_RE.match(self.name):
            raise ValueError(
                f"Invalid skill name '{self.name}'. Must be lowercase letters, numbers, "
                f"and hyphens only. No leading/trailing hyphens. No consecutive hyphens."
            )
        if self.name != self.path.name:
            raise ValueError(
                f"Skill name '{self.name}' must match parent directory '{self.path.name}'."
            )
        return self

    # ------------------------------------------------------------------ #
    #  File access
    # ------------------------------------------------------------------ #

    def list_resources(self) -> list[str]:
        """List bundled resource files (scripts, references, assets) relative to skill root."""
        resources: list[str] = []
        for subdir in ("scripts", "references", "assets"):
            dir_path = self.path / subdir
            if dir_path.is_dir():
                for f in sorted(dir_path.rglob("*")):
                    if f.is_file():
                        resources.append(str(f.relative_to(self.path)).replace("\\", "/"))
        return resources

    def load_reference(self, reference_path: str | Path) -> str:
        """Read a referenced file from within the skill directory."""
        file_path = Path(self.path / reference_path)

        try:
            file_path = file_path.resolve()
            if not str(file_path).startswith(str(self.path.resolve())):
                return f"Access denied: Path traversal detected in {reference_path}"
        except Exception:
            return f"Invalid path: {reference_path}"

        if not file_path.exists():
            return f"File not found: {reference_path}"
        if file_path.is_file():
            if file_path.suffix in SCRIPT_EXTENSIONS | REFERENCE_EXTENSIONS:
                return _safe_read_text(file_path) or f"Failed to read {reference_path}"
            else:
                return f"Unsupported file type. Allowed: {SCRIPT_EXTENSIONS | REFERENCE_EXTENSIONS}"
        else:
            return f"Path is not a file: {reference_path}"

class SkillLoader:
    """
    Skill loader that discovers and parses SKILL.md files.

    Implements the Agent Skills specification progressive disclosure:
      Tier 1 — catalog (name + description) via get_catalog()
      Tier 2 — full instructions via load_skill() / load_all()
      Tier 3 — resource files via Skill.load_reference() / Skill.list_resources()
    """

    def __init__(self, skills_root: str | Path):
        self.skills_root = Path(skills_root)
        if not self.skills_root.exists():
            raise RuntimeError(f"Skills root path does not exist: {self.skills_root}")

        self.logger = logger.bind(component="SkillLoader")

    def get_catalog(self) -> list[dict]:
        """Return Tier 1 catalog: [(name, description)] for all discovered skills.

        Compact enough to embed in a system prompt (~50-100 tokens per skill).
        """
        all_skills = self.load_all()
        return [
            {"name": skill.name, "description": skill.description}
            for skill in all_skills.values()
        ]

    def get_catalog_text(self) -> str:
        """Return Tier 1 catalog as markdown text suitable for system prompts."""
        all_skills = self.load_all()
        if not all_skills:
            return ""
        lines = ["## Available Skills", ""]
        for skill in sorted(all_skills.values(), key=lambda s: s.name):
            lines.append(f"- **{skill.name}**: {skill.description}")
        return "\n".join(lines)


    def load_all(self) -> dict[str, Skill]:
        """
        Load all skills from the skills root directory. Results are cached.

        Returns:
            Dictionary mapping skill_name → Skill.
        """
        if hasattr(self, "_cache"):
            return self._cache

        skills = {}
        duplicate_names = set()

        for skill_path in self.skills_root.iterdir():
            # 安全检查：排除非目录和符号链接
            if not skill_path.is_dir() or skill_path.is_symlink():
                if skill_path.is_symlink():
                    self.logger.warning(f"Skipping symlink: {skill_path}")
                continue

            try:
                skill = self.load_skill(skill_path)
                if skill:
                    # 检测重复名称
                    if skill.name in skills:
                        duplicate_names.add(skill.name)
                        self.logger.warning(
                            f"Duplicate skill name '{skill.name}' detected, "
                            f"existing from {skills[skill.name].path}, "
                            f"new from {skill_path} . Overwriting old one."
                        )
                    else:
                        self.logger.info(f"Loaded skill: {skill.name} from {skill_path}")
                    skills[skill.name] = skill

            except Exception as e:
                self.logger.exception(f"Failed to load skill from {skill_path}", e)

        if duplicate_names:
            self.logger.warning(f"Duplicate skill names found: {duplicate_names}")

        self._cache = skills
        return skills

    def load_skill(self, skill_path: Path) -> Skill | None:
        """
        Load a single skill from a directory.

        Uses lenient validation for cross-client compatibility:
        - Name/directory mismatch → warn, still load
        - Name exceeds 64 chars → warn, still load
        - Missing description → skip (essential for disclosure)
        - Invalid YAML → skip
        """
        skill_file = skill_path / SKILL_FILENAME
        if not skill_file.exists():
            self.logger.warning(f"{SKILL_FILENAME} not found in {skill_path}")
            return None

        content = _safe_read_text(skill_file, self.logger)
        if content is None:
            return None

        metadata, instructions = self._parse_skill_file(content, skill_file)
        if not metadata or not instructions:
            return None

        name = metadata.get("name", "")
        description = metadata.get("description", "")

        # Lenient validation
        if name != skill_path.name:
            self.logger.warning(
                f"Skill name '{name}' does not match directory '{skill_path.name}'. "
                f"Loading anyway for cross-client compatibility."
            )
        if len(name) > 64:
            self.logger.warning(
                f"Skill name '{name}' exceeds 64 characters. Loading anyway."
            )
        if "--" in name:
            self.logger.warning(
                f"Skill name '{name}' contains consecutive hyphens. Loading anyway."
            )

        skill_id = f"{name}@{metadata.get('version', '1.0.0')}"

        allowed_tools = metadata.get("allowed-tools")
        if isinstance(allowed_tools, str):
            allowed_tools = allowed_tools.split()

        skill = Skill(
            id=skill_id,
            name=name,
            description=description,
            path=skill_path,
            instructions=instructions,
            license=metadata.get("license"),
            compatibility=metadata.get("compatibility"),
            metadata=metadata.get("metadata"),
            allowed_tools=allowed_tools,
        )

        return skill

    def _parse_skill_file(self, content: str, file_path: Path) -> tuple[dict | None, str | None]:
        """
        解析SKILL.md文件，分离frontmatter和body

        Args:
            content: 文件内容
            file_path: 文件路径（用于日志）

        Returns:
            (metadata_dict, instructions) 元组
        """
        # 解析YAML frontmatter (--- 包围的部分)
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.search(frontmatter_pattern, content, re.DOTALL)

        if not match:
            self.logger.error(f"Invalid {SKILL_FILENAME} format in {file_path}: missing YAML frontmatter")
            return None, None

        yaml_content = match.group(1)
        instructions = match.group(2).strip()

        try:
            metadata = yaml.safe_load(yaml_content)
            if not isinstance(metadata, dict):
                self.logger.error(f"Invalid YAML frontmatter in {file_path}")
                return None, None

            # 验证必填字段
            if "name" not in metadata or "description" not in metadata:
                self.logger.error(f"Missing required fields in {file_path}: name and description are required")
                return None, None

            return metadata, instructions

        except Exception as e:
            self.logger.exception(f"Failed to parse YAML frontmatter in {file_path}", e)
            return None, None


class SkillContext:
    """Tracks skill activation state for a session (dedup + context protection)."""

    def __init__(self):
        self._activated: dict[str, Skill] = {}

    def is_activated(self, name: str) -> bool:
        return name in self._activated

    def mark_activated(self, skill: Skill):
        self._activated[skill.name] = skill

    def get_activated(self) -> list[Skill]:
        return list(self._activated.values())

    def clear(self):
        self._activated.clear()


skills = SkillLoader(SKILLS_PATH)
skill_context = SkillContext()