"""DABstep OSI SemanticModel factory."""
from __future__ import annotations

from pathlib import Path

from app.semantics.models import SemanticModel, OSISpecification


def build_model() -> SemanticModel:
    """Load the DABstep OSI semantic model from config/benchmark/."""
    yaml_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "benchmark" / "dabstep_model.yaml"
    spec = OSISpecification.load_from_yaml(yaml_path)
    return spec.semantic_model[0]
