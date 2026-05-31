"""Skill activation tool with catalog injection hook."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from app.hook import HookPoint, hook
from app.schema import ToolCall
from app.skill.base import Skill, skills as skill_loader, skill_context
from app.tool.base import BaseTool, ToolResult


def _valid_skill_names() -> list[str]:
    return sorted(skill_loader.load_all().keys())


class SkillActivateTool(BaseTool):
    """Dedicated activation tool — call with a skill name to load its instructions.

    The catalog (available skills with descriptions) is provided in the system
    prompt via get_skill_catalog_text(). When the model identifies a relevant
    skill, it calls this tool with the skill name to load the full SKILL.md body.
    """

    permission: Literal["global", "skills", "agent"] = "agent"
    name: str = Field(default="activate_skill")
    description: str = Field(
        default="Activate a skill by name to load its full instructions into "
        "the conversation. Call this when a task matches a skill listed in "
        "the Available Skills section of the system prompt."
    )
    strict: bool = Field(default=False)
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": _valid_skill_names(),
                    "description": "Name of the skill to activate.",
                }
            },
            "required": ["name"],
        }
    )

    _skills: dict[str, Skill] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _load_skills(self) -> "SkillActivateTool":
        self._skills = skill_loader.load_all()
        return self

    @hook(HookPoint.NODE_INIT_BEFORE)  # callback(ctx, node) — fires once, before first prep
    def _inject_skill_catalog(self, ctx, node):
        catalog = skill_loader.get_catalog_text()
        if not catalog or not node.system_prompt:
            return
        node.system_prompt = (
            node.system_prompt
            + "\n\n<available_skills>\n"
            + catalog
            + "\n</available_skills>"
        )

    def is_available(self) -> bool:
        return len(self._skills) > 0

    async def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        skill_name = tool_call.function.arguments_dict.get("name", "").strip()
        if not skill_name:
            return ToolResult.failure_response(
                tool_call.id, self.name,
                "Missing required parameter 'name'. Specify which skill to activate."
            )

        skill = self._skills.get(skill_name)
        if skill is None:
            available = ", ".join(sorted(self._skills.keys()))
            return ToolResult.failure_response(
                tool_call.id, self.name,
                f"Unknown skill '{skill_name}'. Available: {available}"
            )

        skill_context.mark_activated(skill)

        resources = skill.list_resources()
        resource_lines = ""
        if resources:
            resource_lines = (
                "\n## Bundled Resources\n\n"
                + "\n".join(f"- `{r}`" for r in resources)
                + "\n"
            )

        content = (
            f"<skill_content name=\"{skill.name}\">\n\n"
            f"{skill.instructions}\n\n"
            f"{resource_lines}"
            f"</skill_content>"
        )

        return ToolResult.success_response(tool_call.id, self.name, content)


def has_skills() -> bool:
    """True when at least one skill is available."""
    return len(skill_loader.load_all()) > 0


def create_skill_tool() -> SkillActivateTool | None:
    """Create the activation tool, or None if no skills installed."""
    if not has_skills():
        return None
    return SkillActivateTool()
