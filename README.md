# Stateful Agent Security Evaluation Framework

> Beyond static benchmarks: stateful attack-defense evaluation with uncertainty

A research-grade evaluation framework for testing session-persistent security attacks against LLM-based agents. Addresses the benchmark saturation problem identified in Bhagwatkar et al. (NeurIPS 2025): existing benchmarks are saturated by simple defenses, but none test attacks that survive across conversation resets.

## Three Pillars

1. **Session-persistent attack evaluation** — LangGraph agent with SQLite persistence, multi-session delayed trigger attacks
2. **Obfuscation-based bypass attacks** — Braille, Base64, and semantic indirection against pattern-matching defenses
3. **Statistical rigor** — Bootstrap BCa CIs, power analysis, Holm-Bonferroni corrections, Wilson Score meta-analysis

## Quick Start (Local)

```bash
# 1. Install Ollama (if not already installed)
# Download from https://ollama.ai or use: brew install ollama

# 2. Start Ollama service
ollama serve

# 3. In a new terminal, pull the required models
ollama pull qwen3:8b
ollama pull llama3.1:8b
ollama pull mistral-small3.2:24b

# 4. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 5. Train the Sanitizer classifier
.venv/bin/python scripts/train_sanitizer.py

# 6. Run all tests
.venv/bin/python -m pytest tests/ -q --tb=short

# 7. Dry run (1 run per condition, no stats — verify pipeline works)
.venv/bin/python -m src.runner.runner --dry-run

# 8. Pilot experiment (5 runs × 1 model — verify before full run)
# Edit experiments/configs/pilot.yaml to use 1 model and 5 runs
.venv/bin/python -m src.runner.runner --config experiments/configs/pilot.yaml

# 9. Full factorial experiment (1,080 runs — commit config first!)
git add experiments/configs/default_factorial.yaml
git commit -S -m "pre-register: factorial experiment config"
.venv/bin/python -m src.runner.runner --config experiments/configs/default_factorial.yaml
```

**Note on Ollama setup:**
- Ollama runs on `http://localhost:11434` by default
- To use a remote Ollama instance (e.g., Mac Studio), update `base_url` in the YAML config
- Model weights are cached locally after first pull (~8GB total for all three models)
- All models run with q4_0 quantization for efficiency on consumer hardware

## Quick Start (Docker)

```bash
docker-compose up
```

The docker-compose setup includes:
- Ollama service with persistent volume for model weights
- Python environment with all dependencies
- Shared volume for results and logs

Models are downloaded on first run and cached in the `ollama_models` volume.

## Pre-Run Checklist

Before running the full 1,080-run experiment:

- [ ] Ollama service is running (`ollama serve`)
- [ ] All three models are pulled (`ollama pull qwen3:8b`, etc.)
- [ ] Commit `experiments/configs/default_factorial.yaml` to VCS (pre-registration)
- [ ] Run `scripts/train_sanitizer.py` to generate `data/models/sanitizer_classifier.pkl`
- [ ] Run dry-run mode to verify no crashes
- [ ] Run pilot (5 runs × 1 model) to verify BTCR ≥ 90% on no-attack baseline
- [ ] Check `results/usage_log.json` after pilot for timing estimate
- [ ] Verify cosine similarity between `data/attacks/sensitive_doc.txt` and `data/attacks/malicious_doc.txt` < 0.5 (injection threshold sanity check)

## Project Structure

```
src/
├── agent/          # LangGraph agent + Model_Interface (OpenAI, Anthropic, Ollama)
├── attacks/        # Attack scenarios (delayed trigger, obfuscation bypass, memory poisoning)
├── defenses/       # Defense middleware (Minimizer, Sanitizer, PromptHardening)
├── detection/      # ExfiltrationDetector (3-method OR) + BTCREvaluator
├── runner/         # ExperimentRunner, StateIsolator, ConfigLoader, RateLimiter
├── stats/          # BootstrapEngine (BCa CIs) + MetaAnalyzer (Wilson Score)
└── analysis/       # Plots + LaTeX tables

tests/              # 70 property-based and unit tests (Hypothesis)
experiments/        # YAML configs + results
data/               # Attack payloads, benign context, trained models
scripts/            # train_sanitizer.py
```

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Agent framework | LangGraph + SqliteSaver | Native multi-session checkpointing |
| Defense proxy | TF-IDF + regex + classifier | Lightweight; explicitly NOT the Bhagwatkar LLM-based firewall |
| Meta-analysis CIs | Wilson Score (not bootstrap) | Published papers provide only summary stats |
| Exfiltration detection | Recipient match OR substring OR semantic similarity | Catches direct, copy-paste, and paraphrased leaks |
| Inter-run isolation | Fresh SQLite DB (UUIDv4) per run | Prevents cross-run contamination |

## Scope Limitations

- The Minimizer/Sanitizer defense is a **lightweight proxy**, not the original Bhagwatkar LLM-based firewall. Bypass results are scoped to this proxy.
- Evaluation uses **simulated tools** in a controlled environment. Results may not generalize to production deployments.
- Obfuscation bypass attacks are **static, pre-computed** encodings, not dynamically adaptive attacks.
- Meta-analysis assumes **i.i.d. Bernoulli trials** — published benchmarks may violate this.
- Models are **open-weight Ollama instances** (Qwen 3 8B, Llama 3.1 8B, Mistral Small 3.2 24B). Results may differ from proprietary API models due to quantization, fine-tuning, and inference differences.
