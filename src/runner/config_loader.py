"""Config loader with YAML schema validation (Req 9.6)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

_DATE_PATTERN = re.compile(r"\d{4}-?\d{2}-?\d{2}")
_OLLAMA_VERSION_PATTERN = re.compile(r":\w")

REQUIRED_FIELDS = ["attacks", "defenses", "models", "runs_per_condition", "comparisons"]


@dataclass
class ComparisonSpec:
    condition_a: str
    condition_b: str


@dataclass
class ExperimentConfig:
    attacks: list[dict]
    defenses: list[dict]
    models: list[dict]
    runs_per_condition: int
    comparisons: list[ComparisonSpec]
    effect_size: float = 0.10
    alpha: float = 0.05
    power: float = 0.80
    results_path: str = "results/results.json"
    db_base_dir: str = "data/runs"
    extra: dict = field(default_factory=dict)


def validate_config(config_dict: dict) -> list[str]:
    errors: list[str] = []

    for f in REQUIRED_FIELDS:
        if f not in config_dict:
            errors.append(f"Missing required field: '{f}'")

    # Validate model_name fields
    for model in config_dict.get("models", []):
        name = model.get("model_name", "")
        provider = model.get("provider", "")
        if provider == "ollama":
            if not _OLLAMA_VERSION_PATTERN.search(name):
                errors.append(
                    f"Model '{name}' (ollama) must include a version tag (e.g. 'llama3.1:8b')"
                )
        else:
            if not _DATE_PATTERN.search(name):
                errors.append(
                    f"Model '{name}' must contain a dated version identifier "
                    "(e.g. 'gpt-4o-mini-2024-07-18'). Floating aliases are not allowed."
                )

    # Validate comparisons non-empty
    comparisons = config_dict.get("comparisons", [])
    if isinstance(comparisons, list) and len(comparisons) == 0:
        errors.append("'comparisons' list must be non-empty")

    return errors


def load_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    errors = validate_config(raw)
    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    comparisons = [
        ComparisonSpec(condition_a=c["condition_a"], condition_b=c["condition_b"])
        for c in raw.get("comparisons", [])
    ]

    known = {"attacks", "defenses", "models", "runs_per_condition", "comparisons",
             "effect_size", "alpha", "power", "results_path", "db_base_dir"}
    extra = {k: v for k, v in raw.items() if k not in known}

    return ExperimentConfig(
        attacks=raw["attacks"],
        defenses=raw["defenses"],
        models=raw["models"],
        runs_per_condition=raw["runs_per_condition"],
        comparisons=comparisons,
        effect_size=raw.get("effect_size", 0.10),
        alpha=raw.get("alpha", 0.05),
        power=raw.get("power", 0.80),
        results_path=raw.get("results_path", "results/results.json"),
        db_base_dir=raw.get("db_base_dir", "data/runs"),
        extra=extra,
    )
