"""PersistBench: Benchmark for Persistent Memory Attacks on Stateful LLM Agents.

Evaluates whether LLM agents execute forbidden tool calls when adversarial
payloads are planted in persistent memory at turn N and triggered at turn N+k.

Ground truth: verified tool-call execution (not text heuristics, not LLM judges).
"""

__version__ = "1.0.0"
__benchmark_version__ = "core-v1"
