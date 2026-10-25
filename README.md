# Stateful Agent Security Evaluation

**Paper:** [arXiv:2605.08442](https://arxiv.org/abs/2605.08442)
**Companion:** [Retrieval Observability Bounds on Provenance Detection for Agent Memory Poisoning: Measured Coverage and a Falsified Standalone Detector, arXiv:2606.30566](https://arxiv.org/abs/2606.30566)
**Live site:** [junwenleong.github.io/stateful-agent-security-eval](https://junwenleong.github.io/stateful-agent-security-eval/)

> **Version note (v6).** This repository now describes paper version v6. v6 corrects an argument-order bug in the memory-tool wrapper of the standalone frontier and causal-mediation scripts, which had corrupted the stored value in the earlier frontier collection and in three causal-mediation conditions. The frontier results were re-collected after the fix (29 endpoints, 2026-10-12) and earlier frontier numbers anywhere in this repository's history are superseded. See the paper, [canonical_numbers.md](canonical_numbers.md) and [FINDINGS.md](FINDINGS.md). Known properties of the released harness are listed [below](#known-properties-of-the-released-harness).

Studies the **injection-execution dissociation** in LLM agents with persistent memory: whether a model stores an injected instruction (storage) and whether it later acts on it (execution) are separable outcomes, measured separately. The open-weight factorial evidences the execution stage only, because its nine models were selected for storing the instruction in every run. The frontier evidence is narrower than earlier versions stated: after the fix, most frontier endpoints that store the attacker's rule also execute it, and the three endpoints with no exfiltration under the draft-only trigger (GPT-5.4, GPT-5.6-sol, Claude Haiku 4.5; 0 of 40 runs each) also send almost no email under it. In an exploratory send-authorised follow-up they exfiltrate in 20 of 20, 5 of 20 and 6 of 20 runs, so the frontier data do not show execution resistance.

**Core result:** Storing a directive and acting on it are distinct events, and a zero attack rate is not evidence of safety when the agent never reaches a send decision. Of 29 frontier endpoints (N=10 each, four topped up to N=40), 7 exfiltrate in at least 9 of 10 runs and 11 more in some but fewer; several Anthropic endpoints, GPT-5.6-terra, GPT-5.6-luna and Gemini 3.7 Flash do not store the attacker's address. OpenAI snapshots show no monotonic ordering. In the factorial, five of six defences fail, for different reasons (input-layer filters never see a payload that arrives by retrieval; the small retrieval-layer classifiers tested do not flag it; prompt hardening is overridden). Only Memory Sandbox, which removes the recall tool, reduces ASR to 0% for 8/9 models, and it is bypassed when a model rebuilds the instruction from the retrieval corpus (RAG fallback). All frontier results describe hosted endpoints during one collection window. Full results in [FINDINGS.md](FINDINGS.md), technical writeup at the [live site](https://junwenleong.github.io/stateful-agent-security-eval/).

---

> Beyond static benchmarks: stateful attack-defense evaluation with uncertainty

A research-grade evaluation framework for testing session-persistent security attacks against LLM-based agents. Addresses the benchmark saturation problem: existing benchmarks test only input-level attacks, but none test attacks that survive across conversation resets via persistent memory.

## Three Pillars

1. **Injection-execution dissociation** - Measuring storage and execution as separate outcomes across models, with the stage at which each model stops the attack reported per endpoint
2. **Layer-structural defense analysis** - Tool-call instrumentation comparing defences by the layer at which they act (input, retrieval, instruction, tool), and by whether they can observe the payload
3. **Statistical reporting** - 5,040-run factorial with bootstrap BCa CIs, Holm-Bonferroni corrections and Wilson Score intervals; most factorial cells are all-success or all-failure, so the counts matter more than the p-values

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

# 7. Dry run (1 run per condition - verify pipeline works)
.venv/bin/python scripts/run_defense_factorial.py --dry-run

# 8. Full factorial (9 models, ~8.5 days total on a single GPU)
.venv/bin/python scripts/run_defense_factorial.py
```

**Ollama settings for the factorial:**
- `OLLAMA_NUM_PARALLEL=1` - one request at a time (prevents queue buildup)
- `OLLAMA_MAX_LOADED_MODELS=1` - prevents OOM on large model transitions (72+75GB)
- `OLLAMA_KEEP_ALIVE=5m` - keeps model warm between sequential runs
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
├── run_defense_factorial.py   # Main entrypoint - phased execution
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
| Session isolation | Fresh thread_id per session | Enforces context wipe - attack must be tool-mediated, not context-leaked |
| Defense proxy | TF-IDF + regex + classifier | Lightweight; explicitly NOT an LLM-based firewall |
| Meta-analysis CIs | Wilson Score (not bootstrap) | Published papers provide only summary stats |
| Exfiltration detection | Recipient match OR substring OR semantic similarity | Catches direct, copy-paste, and paraphrased leaks |
| Inter-run isolation | Fresh SQLite DB (UUIDv4) per run | Prevents cross-run contamination |
| Concurrency | workers=1, OLLAMA_NUM_PARALLEL=1 | Single GPU bottleneck; parallel workers add swap overhead without speedup |

## Requirements

- Python 3.11 (required - `python3.11 -m venv .venv`)
- Ollama (for open-source models)
- AWS credentials with Bedrock access (for frontier model screening only)

## Data Download (Git LFS)

The 5,040-run factorial log (`results/defense_factorial/results.jsonl`, about 120 MB) is stored with Git LFS. A plain `git clone` fetches LFS files automatically when `git-lfs` is installed (`brew install git-lfs && git lfs install`); otherwise the file is a three-line pointer. To clone without it, use `GIT_LFS_SKIP_SMUDGE=1 git clone ...`, and fetch it later with `git lfs pull`. GitHub applies a free bandwidth quota to LFS downloads, so a fetch can fail when the quota is exhausted; retry later or contact the author.

## Factorial Design

**9 models × 7 conditions (6 defenses + baseline) × 2 attacks × N=40 = 5,040 runs**

Models: `qwen2.5:14b`, `qwen3.5:9b`, `qwen3:32b`, `qwen2.5:72b`, `qwen3.5:122b`, `qwq:32b`, `glm-4.7-flash:q8_0`, `gpt-oss:20b`, `gpt-oss-safeguard:120b`

Defenses: `no_defense`, `minimizer`, `sanitizer`, `rag_sanitizer`, `prompt_hardening`, `memory_sandbox`, `rag_llm_judge`

Attacks: `delayed_trigger`, `no_attack` (baseline)

Phases (sequential, one model at a time to manage VRAM):
- Phases 1-9: one model per phase, N=560 runs each (~0.3-2.0 days depending on model size)
- Total wall time: ~8.5 days on a single GPU (Mac Studio M3 Ultra, 96GB unified memory)

## Scope Limitations

- Input-level defences (Minimiser, Sanitiser) are **lightweight proxies** (TF-IDF + a classifier trained on 48 of 60 examples, 12 held out), but the input-layer finding is architectural: production input-layer classifiers (Llama Guard, ShieldGemma) deployed only on the user message share the same blind spot, because they cannot observe RAG-retrieved content. The retrieval-layer result is narrower: the small classifiers and judges tested (1.5B, 7B) do not flag the payload, and three frontier-model judges did flag it at document level (they were not run as an agent defence, and the test used one malicious template).
- Evaluation uses **simulated tools** in a controlled environment. Results may not generalise to production deployments.
- The factorial models are **open-weight Ollama instances** (q4_0 or q8_0 quantisation). Results may differ from full-precision or API-served versions. Frontier results come from hosted endpoints under their requested names, with no provider-side confirmation of the served build.
- The RAG corpus is **in-memory** (three documents; every query returns all of them) and is not a cloud vector store. Infrastructure is held constant across models by design so that differences between models are not infrastructure differences. The attack is fixed and non-adaptive, so the rates are observed rates under that attack, not worst-case rates.
- Memory Sandbox's "no utility cost" is a statement about the single-session no-attack control, which does not require recall; utility for memory-dependent workflows was not evaluated, and under attack two models stop attempting the task after recall removal.
- The no-attack arm is a no-false-trigger check, not a false-positive-rate measurement: its detector had no attacker address to match.
- See "Known properties of the released harness" below (for example, the benign calendar and search tools return empty results in the factorial).

## Known properties of the released harness

The data in this repository were collected with the behaviour below. It is left unchanged so that the released data stay reproducible; none of it changes a headline 0% or 100% factorial cell, but readers should know it.

- **Benign fixtures are cleared.** The runner loads the benign calendar and search fixtures and then calls the tool reset, which clears them. Calendar and search tools therefore return empty results in the factorial (every calendar and search call in the logs has `count: 0`).
- **Most RAG queries return the whole corpus.** The corpus holds three documents and the model chooses `top_k` (default 3); 7,536 of the 7,816 factorial RAG queries asked for three or more and so returned all of them, including the malicious document, in any session in which the agent queries RAG.
- **RATG exposes an extra tool.** With RATG enabled, the framework also advertises a `memory_ratg_filter` tool to the model (10 tools become 11). It was never called in any collected RATG log.
- **The injection flag is not bounded to S1.** `evaluate_injection` counts a `memory_save_fact` call anywhere after the injection-session log offset. On a strict S1 reading, 2,517 of 2,520 attack-arm runs have a save in S1; ASR is unaffected.
- **Mechanistic tags are outcome-derived labels.** The deterministic tags (for example `explicit_recall_required`: sandbox arm, a save, no exfiltration) are computed from outcomes. They are descriptive labels, not independent trace evidence of a mechanism.
- **The local LLM judge fails open.** On a timeout, HTTP error or parse failure it returns "not malicious".
- **The Ollama wrapper does not forward assistant `tool_calls`** in the conversation history (tool results are forwarded). This applies to every arm.
- **Frontier scripts score requested sends.** Attack success is scored from the recipients of the requested `email_send_email` calls, and send results are not logged, so the frontier rate is an attacker-addressed send rate. Frontier records carry the requested model name, not a resolved build.
- **Some state persists across sessions in the local runner.** The email send counter and the RAG corpus persist across sessions, and the agent can call `email_get_emails` and `rag_inject_document` (used in 3 of 5,040 runs and 0 runs respectively), so SQLite memory is not the only cross-session state.
- **Digest pinning is optional.** The model-digest guard exists but is not set in the factorial configuration, and the 5,040 factorial records carry no model hash.
- **Docker.** The supplied Dockerfile provides the Python environment with the source, configurations and scripts (including the verification script). It does not copy the result data, and its default command exits without running an experiment, so it is not a one-command reproduction.

## Ethics and Responsible Disclosure

This research evaluates defense effectiveness against a known class of attacks (persistent memory injection) using **simulated tools in a sandboxed environment**. No attacks were executed against production systems, real users, or live vendor APIs beyond standard model inference calls.

**Scope of published materials:**
- Attack payloads use fictional compliance frameworks (SEC Rule 17a-4 / FSCF-2024 framing) and fictional email addresses
- All tools are simulated (no real emails sent, no real databases accessed)
- The evaluation framework is designed for controlled research, not weaponization

**Vendor notification:** No vendor was notified of these results before publication. (An earlier version of the paper stated that disclosure had been made; that statement was wrong and is withdrawn.) Frontier model results were obtained through standard API access. The findings document behavioural properties observable through normal inference, not exploitation of undisclosed vulnerabilities. The attack vector (RAG corpus poisoning into persistent memory) is a known threat class documented in concurrent literature.

**Intended use:** This repository is intended for security researchers evaluating defense mechanisms against persistent memory attacks. If you identify a vulnerability in a production system using these techniques, please follow responsible disclosure practices with the affected vendor.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{leong2026dissociation,
  title={Injection-Execution Dissociation: A Mechanistic Evaluation of Persistent Memory Attacks and Defences in Stateful LLM Agents},
  author={Leong, Jun Wen},
  journal={arXiv preprint arXiv:2605.08442},
  year={2026}
}
```

Companion forensic detection paper:

```bibtex
@article{leong2026trajectory,
  title={Retrieval Observability Bounds on Provenance Detection for Agent Memory Poisoning: Measured Coverage and a Falsified Standalone Detector},
  author={Leong, Jun Wen},
  journal={arXiv preprint arXiv:2606.30566},
  year={2026}
}
```