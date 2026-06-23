# Stateful Agent Security Evaluation

**Paper:** [arXiv:2605.08442](https://arxiv.org/abs/2605.08442)

**Five of six defences fail completely against delayed trigger attacks that persist through LLM agent memory. Tested across 5,040 runs, 9 models, 6 defences + undefended baseline.**

Additional findings: the only defence that works inverted to 100% attack success on one model under Ollama 0.20.6 (an inference-engine-specific artifact — the same model exfiltrates under no defence on newer Ollama versions, eliminating the inversion); prompt hardening can accelerate attacks via RAG re-injection; a safety fine-tuned model achieves 100% ASR; one frontier model stores security alerts instead of payloads; a latent carrier model persists financial credentials without attacker instruction.

**v2 addition (June 2026):** A reasoning-mode ablation reveals a double dissociation: the sandbox variant protecting reasoning models collapses the attack for non-reasoning models (session 3 → session 0), while the variant protecting non-reasoning models is bypassed by reasoning models via goal-directed RAG fallback. No single memory-sandbox implementation is safe across both model classes. The qwq:32b Draft-Only archetype observed in the factorial (Ollama 0.20.6) is inference-engine-sensitive and does not reproduce under Ollama 0.21.2 — the original inversion finding is reclassified as an inference-engine artifact rather than a stable model property.

Full results and methodology in [FINDINGS.md](FINDINGS.md). Technical writeup at [junwenleong.github.io/stateful-agent-security-eval](https://junwenleong.github.io/stateful-agent-security-eval/).

---

> Beyond static benchmarks: stateful attack-defense evaluation with uncertainty

A research-grade evaluation framework for testing session-persistent security attacks against LLM-based agents. Addresses the benchmark saturation problem: existing benchmarks test only input-level attacks, but none test attacks that survive across conversation resets via persistent memory.

## Three Pillars

1. **Session-persistent attack evaluation** — LangGraph agent with SQLite persistence, multi-session delayed trigger attacks
2. **Mechanistic defense analysis** — Tool-call instrumentation distinguishing injection-stage vs. execution-stage blocking
3. **Statistical rigor** — Bootstrap BCa CIs, power analysis, Holm-Bonferroni corrections, Wilson Score meta-analysis

## Quick Start (Local)

```bash
# 1. Install Ollama
# Download from https://ollama.ai or: brew install ollama

# 2. Start Ollama with factorial settings
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=5m ollama serve

# 3. Pull the models used in the factorial
ollama pull qwen2.5:14b
ollama pull qwen3.5:9b
ollama pull qwen3:32b
ollama pull qwen2.5:72b
ollama pull qwen3.5:122b
ollama pull qwq:32b
ollama pull glm-4.7-flash:q8_0
ollama pull gpt-oss:20b
ollama pull gpt-oss-safeguard:120b

# 4. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 5. Train the Sanitizer classifier
.venv/bin/python scripts/train_sanitizer.py

# 6. Run all tests
.venv/bin/python -m pytest tests/ -q --tb=short

# 7. Dry run (1 run per condition — verify pipeline works)
.venv/bin/python scripts/run_defense_factorial.py --dry-run

# 8. Full factorial (9 models, ~8.5 days total on a single GPU)
.venv/bin/python scripts/run_defense_factorial.py
```

**Ollama settings for the factorial:**
- `OLLAMA_NUM_PARALLEL=1` — one request at a time (prevents queue buildup)
- `OLLAMA_MAX_LOADED_MODELS=1` — prevents OOM on large model transitions (72+75GB)
- `OLLAMA_KEEP_ALIVE=5m` — keeps model warm between sequential runs
- Models run with q4_0 quantization

> **These env vars are required, not optional.** Omitting `OLLAMA_MAX_LOADED_MODELS=1` during the Phase 3→4 transition (qwen3.5:122b 75GB → gpt-oss-safeguard 72GB) risks OOM and a silent daemon crash. Omitting `OLLAMA_NUM_PARALLEL=1` causes request queue buildup that triggers the 600s HTTP timeout on slower models.

## Pre-Run Checklist

Before running the full factorial:

- [ ] Ollama service is running with correct env vars (see above)
- [ ] Required models are pulled (`ollama list` to verify)
- [ ] `scripts/train_sanitizer.py` has been run to generate `data/models/sanitizer_classifier.pkl`
- [ ] Dry run completes with 0 errors (`--dry-run --phase 1`)
- [ ] BTCR ≥ 90% on no-attack baseline in dry run
- [ ] Results directory exists: `results/defense_factorial/`

## Project Structure

```
src/
├── agent/          # LangGraph agent + ModelInterface (OpenAI, Anthropic, Ollama, Bedrock)
├── attacks/        # Attack scenarios (delayed_trigger, no_attack)
├── defenses/       # Defense middleware (Minimizer, Sanitizer, RAGSanitizer, PromptHardening, MemorySandbox, RAGLLMJudge)
├── detection/      # ExfiltrationDetector (3-method OR) + BTCREvaluator
├── runner/         # ExperimentRunner, StateIsolator, ConfigLoader, ParallelRunner
├── stats/          # BootstrapEngine (BCa CIs) + MetaAnalyzer (Wilson Score)
└── analysis/       # Plots, LaTeX tables, MechanisticAnalyzer

scripts/
├── run_defense_factorial.py   # Main entrypoint — phased execution
├── run_bedrock_apac_smoke.py  # Bedrock frontier smoke test
├── run_haiku_memory_sandbox.py # Haiku supplementary evaluation
├── run_n10_all_models.py      # N=10 screening across all models
├── analyze_results.py         # BCa bootstrap analysis + Holm-Bonferroni
├── n10_analysis.py            # N=10 archetype classification
├── train_sanitizer.py         # Train TF-IDF sanitizer classifier
└── verify_canonical.py        # Programmatic verification of all published numbers

experiments/configs/
├── defense_factorial.yaml     # Main factorial config (9 models × 7 conditions × 2 attacks)
├── bedrock_apac_smoke.yaml    # Bedrock frontier model config
├── haiku_memory_sandbox.yaml  # Haiku supplementary evaluation config
└── n10_all_models.yaml        # N=10 screening config

tests/              # Property-based and unit tests (Hypothesis + pytest)
data/               # Attack payloads, benign context, trained models
results/            # Experiment outputs (JSONL + run logs)
```

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Agent framework | LangGraph + SqliteSaver | Native multi-session checkpointing |
| Session isolation | Fresh thread_id per session | Enforces context wipe — attack must be tool-mediated, not context-leaked |
| Defense proxy | TF-IDF + regex + classifier | Lightweight; explicitly NOT an LLM-based firewall |
| Meta-analysis CIs | Wilson Score (not bootstrap) | Published papers provide only summary stats |
| Exfiltration detection | Recipient match OR substring OR semantic similarity | Catches direct, copy-paste, and paraphrased leaks |
| Inter-run isolation | Fresh SQLite DB (UUIDv4) per run | Prevents cross-run contamination |
| Concurrency | workers=1, OLLAMA_NUM_PARALLEL=1 | Single GPU bottleneck; parallel workers add swap overhead without speedup |

## Requirements

- Python 3.11 (required — `python3.11 -m venv .venv`)
- Ollama (for open-source models)
- AWS credentials with Bedrock access (for frontier model screening only)

## Factorial Design

**9 models × 7 conditions (6 defenses + baseline) × 2 attacks × N=40 = 5,040 runs**

Models: `qwen2.5:14b`, `qwen3.5:9b`, `qwen3:32b`, `qwen2.5:72b`, `qwen3.5:122b`, `qwq:32b`, `glm-4.7-flash:q8_0`, `gpt-oss:20b`, `gpt-oss-safeguard:120b`

Defenses: `no_defense`, `minimizer`, `sanitizer`, `rag_sanitizer`, `prompt_hardening`, `memory_sandbox`, `rag_llm_judge`

Attacks: `delayed_trigger`, `no_attack` (baseline)

Phases (sequential, one model at a time to manage VRAM):
- Phases 1–9: one model per phase, N=560 runs each (~0.3–2.0 days depending on model size)
- Total wall time: ~8.5 days on a single GPU (Mac Studio M2 Ultra, 192GB unified memory)

## Scope Limitations

- The Minimizer/Sanitizer defense is a **lightweight proxy** (TF-IDF + 60-example classifier). Results are scoped to this proxy, not production-grade defenses.
- Evaluation uses **simulated tools** in a controlled environment. Results may not generalize to production deployments.
- Models are **Ollama instances** with q4_0 quantization. Results may differ from full-precision or API-served versions.
- Meta-analysis assumes **i.i.d. Bernoulli trials** — published benchmarks may violate this.
- The RAG corpus is **in-memory** (not a cloud vector store). Infrastructure is held constant across models by design to isolate model reasoning as the independent variable.
