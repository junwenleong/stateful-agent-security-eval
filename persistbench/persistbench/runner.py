"""Main orchestrator for PersistBench runs.

Loads scenarios, runs attacks, applies defenses, executes agent interactions,
and verifies results via the tool-call oracle.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from persistbench.adapters.base import Message, ModelAdapter, ModelResponse
from persistbench.attacks.authority_spoofing import AuthoritySpoofingAttack
from persistbench.attacks.base import Attack
from persistbench.attacks.direct_injection import DirectInjectionAttack
from persistbench.attacks.reasoning_mode import ReasoningModeAttack
from persistbench.config import BenchmarkConfig, ScenarioConfig
from persistbench.defenses.base import (
    ConversationIsolation,
    Defense,
    InputValidation,
    NoDefense,
    PromptReinforcement,
    ToolSchemaRestriction,
)
from persistbench.defenses.memory_sandbox import MemorySandbox
from persistbench.memory import PersistentMemoryStore
from persistbench.registry import ToolRegistry
from persistbench.verifiers.tool_call import ToolCallVerifier, VerificationResult


ATTACK_REGISTRY: dict[str, type[Attack]] = {
    "authority_spoofing": AuthoritySpoofingAttack,
    "direct_injection": DirectInjectionAttack,
    "reasoning_mode": ReasoningModeAttack,
}

DEFENSE_REGISTRY: dict[str, type[Defense]] = {
    "none": NoDefense,
    "input_validation": InputValidation,
    "prompt_reinforcement": PromptReinforcement,
    "conversation_isolation": ConversationIsolation,
    "tool_restriction": ToolSchemaRestriction,
    "memory_sandbox": MemorySandbox,
}


@dataclass
class RunResult:
    """Result of a single scenario run."""
    verification: VerificationResult
    duration_seconds: float
    turns_executed: int
    memory_state: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class BenchmarkRunner:
    """Orchestrates benchmark execution."""

    def __init__(
        self,
        adapter: ModelAdapter,
        config: BenchmarkConfig | None = None,
        output_dir: str | Path = "results",
        verbose: bool = False,
    ) -> None:
        self._adapter = adapter
        self._config = config
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._verbose = verbose
        self._verifier = ToolCallVerifier()
        self._results: list[RunResult] = []

    def run_scenario(self, scenario: ScenarioConfig) -> RunResult:
        """Run a single scenario end-to-end."""
        start_time = time.time()

        # Initialize components
        memory = PersistentMemoryStore()
        tools = ToolRegistry(memory)

        # Instantiate attack
        attack = self._create_attack(scenario.attack, scenario.attack_params)

        # Plant the attack payload (turn 0 - injection phase)
        attack_result = attack.plant(memory)
        memory.snapshot_initial_state()

        # Instantiate and apply defense
        defense = self._create_defense(scenario.defense, scenario.defense_params)
        tool_schemas = tools.get_all_schemas()
        memory, system_prompt, tool_schemas, defense_result = defense.apply(
            memory, scenario.system_prompt, tool_schemas)

        # Build initial messages
        messages: list[Message] = [Message(role="system", content=system_prompt)]

        # Inject memory content into context (simulating memory recall at session start)
        memory_content = self._format_memory_for_context(memory)
        if memory_content:
            messages.append(Message(
                role="system",
                content=f"[Persistent Memory Contents]\n{memory_content}\n[End Memory]",
            ))

        # Execute trigger turn
        memory.advance_turn()
        tools.set_turn(memory.current_turn)

        trigger_msg = attack.get_trigger_message()
        messages.append(Message(role="user", content=trigger_msg))

        # Agent interaction loop (handle tool calls)
        turns_executed = 0
        max_iterations = 10  # Safety cap

        for _ in range(max_iterations):
            turns_executed += 1
            response = self._adapter.chat(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                temperature=self._config.temperature if self._config else 0.0,
                max_tokens=self._config.max_tokens if self._config else 1024,
            )

            if response.has_tool_calls:
                # Record assistant message with tool calls
                messages.append(Message(
                    role="assistant", content=response.content,
                    tool_calls=response.tool_calls,
                ))
                # Execute each tool call
                for tc in response.tool_calls:
                    result = tools.execute_tool_call(tc.function_name, tc.arguments)
                    messages.append(Message(
                        role="tool", content=result,
                        tool_call_id=tc.id, name=tc.function_name,
                    ))
            else:
                # Final text response, no more tool calls
                messages.append(Message(role="assistant", content=response.content))
                break

        # Verify results
        verification = self._verifier.verify(
            scenario_id=scenario.id,
            attack_name=attack.name,
            defense_name=defense.name,
            model_id=self._adapter.model_id(),
            invocations=tools.get_all_invocations(),
            forbidden_functions=attack.get_forbidden_calls(),
            trigger_turn=scenario.trigger_turn,
        )

        duration = time.time() - start_time
        run_result = RunResult(
            verification=verification,
            duration_seconds=duration,
            turns_executed=turns_executed,
            memory_state=memory.to_dict(),
        )
        self._results.append(run_result)

        if self._verbose:
            status = "ATTACK SUCCESS" if verification.attack_success else "DEFENDED"
            print(f"  [{status}] {scenario.id} | {attack.name} vs {defense.name} | {duration:.1f}s")

        return run_result

    def run_all(self, scenarios: list[ScenarioConfig] | None = None) -> list[RunResult]:
        """Run all scenarios in the benchmark."""
        if scenarios is None and self._config:
            scenarios = self._config.scenarios
        if not scenarios:
            raise ValueError("No scenarios to run")

        results = []
        for i, scenario in enumerate(scenarios, 1):
            if self._verbose:
                print(f"[{i}/{len(scenarios)}] Running: {scenario.id}")
            result = self.run_scenario(scenario)
            results.append(result)
        return results

    def save_results(self, filename: str = "results.jsonl") -> Path:
        """Save all results to JSONL."""
        output_path = self._output_dir / filename
        with open(output_path, "w") as f:
            for result in self._results:
                record = {
                    **result.verification.to_dict(),
                    "duration_seconds": result.duration_seconds,
                    "turns_executed": result.turns_executed,
                    "error": result.error,
                }
                f.write(json.dumps(record) + "\n")
        return output_path

    def _create_attack(self, name: str, params: dict[str, Any]) -> Attack:
        attack_cls = ATTACK_REGISTRY.get(name)
        if attack_cls is None:
            raise ValueError(f"Unknown attack: {name}. Available: {list(ATTACK_REGISTRY.keys())}")
        return attack_cls(**params)

    def _create_defense(self, name: str, params: dict[str, Any]) -> Defense:
        defense_cls = DEFENSE_REGISTRY.get(name)
        if defense_cls is None:
            raise ValueError(f"Unknown defense: {name}. Available: {list(DEFENSE_REGISTRY.keys())}")
        return defense_cls(**params)

    def _format_memory_for_context(self, memory: PersistentMemoryStore) -> str:
        """Format memory entries for injection into conversation context."""
        entries = memory.get_all_entries(namespace="default")
        if not entries:
            return ""
        lines = []
        for entry in entries:
            lines.append(f"- {entry.key}: {entry.value}")
        return "\n".join(lines)
