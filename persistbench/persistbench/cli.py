"""CLI entry point for PersistBench.

Usage:
    persistbench run --profile smoke|core|full --model MODEL [--attack ATTACK] [--defense DEFENSE]
    persistbench report results.jsonl
    persistbench list-models
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from persistbench import __version__


def get_adapter(model: str):
    """Create the appropriate adapter for the given model string."""
    if model == "recorded" or model.startswith("recorded/"):
        from persistbench.adapters.recorded_adapter import RecordedAdapter
        trace_dir = Path(__file__).parent.parent / "benchmarks" / "core-v1" / "traces"
        trace_file = trace_dir / "smoke_authority_spoofing.jsonl"
        return RecordedAdapter(trace_file, model_name="smoke-recorded")

    if model.startswith("ollama/"):
        from persistbench.adapters.ollama_adapter import OllamaAdapter
        return OllamaAdapter(model=model.removeprefix("ollama/"))

    if "claude" in model or "anthropic" in model:
        from persistbench.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(model=model)

    # Default: OpenAI-compatible
    from persistbench.adapters.openai_adapter import OpenAIAdapter
    return OpenAIAdapter(model=model)


def cmd_run(args: argparse.Namespace) -> int:
    """Run the benchmark."""
    from persistbench.config import load_benchmark
    from persistbench.reporting import generate_report, save_report
    from persistbench.runner import BenchmarkRunner

    # Determine benchmark manifest
    bench_dir = Path(__file__).parent.parent / "benchmarks" / "core-v1"
    manifest_path = bench_dir / "benchmark.yaml"

    if not manifest_path.exists():
        print(f"ERROR: Benchmark manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    config = load_benchmark(manifest_path)

    # Filter scenarios by attack/defense if specified
    scenarios = config.scenarios
    if args.attack:
        scenarios = [s for s in scenarios if s.attack == args.attack]
    if args.defense:
        scenarios = [s for s in scenarios if s.defense == args.defense]

    if not scenarios:
        print("ERROR: No scenarios match the given filters.", file=sys.stderr)
        return 1

    # Profile handling
    if args.profile == "smoke":
        # Smoke profile: use recorded adapter, no API calls
        adapter = get_adapter("recorded")
        print(f"PersistBench v{__version__} | Profile: smoke (offline)")
        print(f"Using recorded traces - no API calls needed")
    else:
        if not args.model:
            print("ERROR: --model required for core/full profiles.", file=sys.stderr)
            return 1
        adapter = get_adapter(args.model)
        if not adapter.is_available():
            print(f"ERROR: Model '{args.model}' not available (missing credentials?).", file=sys.stderr)
            return 1
        print(f"PersistBench v{__version__} | Profile: {args.profile}")
        print(f"Model: {adapter.model_id()}")

    print(f"Scenarios: {len(scenarios)} | Repetitions: {config.repetitions}")
    print("-" * 50)

    # Run
    output_dir = Path(args.output) if args.output else Path("results") / args.profile
    runner = BenchmarkRunner(adapter=adapter, config=config, output_dir=output_dir, verbose=True)

    start = time.time()
    all_results = []
    for rep in range(config.repetitions):
        if config.repetitions > 1:
            print(f"\n--- Repetition {rep + 1}/{config.repetitions} ---")
        results = runner.run_all(scenarios)
        all_results.extend(results)
    duration = time.time() - start

    # Report
    verifications = [r.verification for r in all_results]
    report = generate_report(verifications, duration)
    save_report(report, output_dir)
    results_path = runner.save_results()

    print("\n" + report.summary())
    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to: {output_dir}/report.json")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate report from existing results."""
    from persistbench.reporting import generate_report, save_report
    from persistbench.verifiers.tool_call import VerificationResult

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}", file=sys.stderr)
        return 1

    results = []
    with open(results_path) as f:
        for line in f:
            data = json.loads(line)
            results.append(VerificationResult(
                scenario_id=data["scenario_id"],
                attack_name=data["attack_name"],
                defense_name=data["defense_name"],
                model_id=data["model_id"],
                attack_success=data["attack_success"],
                total_tool_calls=data.get("total_tool_calls", 0),
                trigger_turn=data.get("trigger_turn", 0),
            ))

    report = generate_report(results)
    print(report.summary())

    output_dir = results_path.parent
    save_report(report, output_dir)
    print(f"\nReport saved to: {output_dir}/report.json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="persistbench",
        description="PersistBench: Benchmark for Persistent Memory Attacks on Stateful LLM Agents",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # run command
    run_parser = subparsers.add_parser("run", help="Run the benchmark")
    run_parser.add_argument("--profile", choices=["smoke", "core", "full"], default="smoke")
    run_parser.add_argument("--model", type=str, default=None, help="Model to evaluate")
    run_parser.add_argument("--attack", type=str, default=None, help="Filter by attack type")
    run_parser.add_argument("--defense", type=str, default=None, help="Filter by defense type")
    run_parser.add_argument("--output", type=str, default=None, help="Output directory")

    # report command
    report_parser = subparsers.add_parser("report", help="Generate report from results")
    report_parser.add_argument("results_file", help="Path to results.jsonl")

    args = parser.parse_args()
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "report":
        return cmd_report(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
