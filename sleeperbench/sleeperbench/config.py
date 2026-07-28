"""Configuration loading from YAML benchmark manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScenarioConfig:
    """Configuration for a single benchmark scenario."""
    id: str
    attack: str
    attack_params: dict[str, Any] = field(default_factory=dict)
    defense: str = "none"
    defense_params: dict[str, Any] = field(default_factory=dict)
    system_prompt: str = "You are a helpful assistant with access to tools."
    trigger_turn: int = 1
    description: str = ""


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration."""
    name: str
    version: str
    description: str
    scenarios: list[ScenarioConfig]
    models: list[str] = field(default_factory=list)
    repetitions: int = 1
    temperature: float = 0.0
    max_tokens: int = 1024

    @property
    def total_runs(self) -> int:
        return len(self.scenarios) * max(len(self.models), 1) * self.repetitions


def load_benchmark(manifest_path: str | Path) -> BenchmarkConfig:
    """Load a benchmark configuration from a YAML manifest."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    scenarios = []
    for s in data.get("scenarios", []):
        if isinstance(s, str):
            # Reference to external scenario file
            scenario_path = path.parent / s
            with open(scenario_path) as sf:
                s = yaml.safe_load(sf)

        scenarios.append(ScenarioConfig(
            id=s["id"],
            attack=s["attack"],
            attack_params=s.get("attack_params", {}),
            defense=s.get("defense", "none"),
            defense_params=s.get("defense_params", {}),
            system_prompt=s.get("system_prompt", "You are a helpful assistant with access to tools."),
            trigger_turn=s.get("trigger_turn", 1),
            description=s.get("description", ""),
        ))

    return BenchmarkConfig(
        name=data["name"],
        version=data["version"],
        description=data.get("description", ""),
        scenarios=scenarios,
        models=data.get("models", []),
        repetitions=data.get("repetitions", 1),
        temperature=data.get("temperature", 0.0),
        max_tokens=data.get("max_tokens", 1024),
    )


def load_scenario(scenario_path: str | Path) -> ScenarioConfig:
    """Load a single scenario from a YAML file."""
    path = Path(scenario_path)
    with open(path) as f:
        s = yaml.safe_load(f)
    return ScenarioConfig(
        id=s["id"],
        attack=s["attack"],
        attack_params=s.get("attack_params", {}),
        defense=s.get("defense", "none"),
        defense_params=s.get("defense_params", {}),
        system_prompt=s.get("system_prompt", "You are a helpful assistant with access to tools."),
        trigger_turn=s.get("trigger_turn", 1),
        description=s.get("description", ""),
    )


PROFILES: dict[str, str] = {
    "smoke": "Uses recorded traces, no API calls, <10min",
    "core": "Representative subset, ~200 runs per model",
    "full": "Complete benchmark, all 5,040 runs",
}
