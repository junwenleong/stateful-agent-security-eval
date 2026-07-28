# Artifact Appendix: SleeperBench

## Abstract

This artifact accompanies the paper "Persistent Memory Attacks on Stateful LLM Agents:
A Cross-Vendor Defense Evaluation." It provides:

1. **SleeperBench v1.0.0** — an open-source benchmark for evaluating persistent memory
   attacks against stateful LLM agents, with 5 defense layers and ground-truth
   tool-call verification.

2. **Raw experimental data** — all 5,040 pre-registered factorial trials (9 models ×
   5 attack variants × 7 defense layers × 10 repetitions + supplementary runs),
   plus all follow-up experiments cited in the paper.

3. **Reproduction scripts** — automated pipelines that regenerate all tables and
   figures from the raw data, with validation against canonical numbers.

## Artifact Identification

- **Paper title:** Persistent Memory Attacks on Stateful LLM Agents: A Cross-Vendor Defense Evaluation
- **Submission ID:** [To be assigned]
- **Artifact DOI:** [To be assigned upon acceptance]
- **Repository:** Anonymous during review; will be published at acceptance.

## Badges Claimed

We claim all three NDSS artifact badges:

### Available

- Complete source code for SleeperBench v1.0.0 (22 Python source files, MIT license)
- All raw experimental data (JSONL format, ~50MB uncompressed)
- All analysis scripts
- Hosted at [anonymous repository URL]

### Functional

- SleeperBench installs via `pip install -e ".[all]"` on Python 3.11-3.13
- Smoke test runs offline in <2 minutes with no API keys
- Core benchmark reproduces representative subset in ~30 minutes per model
- All dependencies pinned in `pyproject.toml`

### Reproduced

- `scripts/reproduce_paper.sh` regenerates all paper tables/figures from raw data
- `scripts/verify_canonical.py` validates every number in the paper against raw data
- Key claims independently reproducible with API keys:
  - 88.9% overall ASR on failing defenses (320/360 runs)
  - Memory Sandbox: 0% ASR on 8/9 models
  - Reasoning-mode bypass: 100% on qwq:32b
  - Tripartite vendor pattern

## Environment Requirements

### Hardware
- Any machine with 8+ GB RAM and internet access
- Apple Silicon Mac recommended for local-model experiments (Ollama)
- No GPU required for API-based experiments

### Software
- Python 3.11, 3.12, or 3.13
- pip (any recent version)
- Git

### API Keys (for full reproduction)
- `OPENAI_API_KEY` — for GPT-4o, GPT-4o-mini experiments
- `ANTHROPIC_API_KEY` — for Claude experiments
- `OLLAMA_HOST` — for local model experiments (optional; requires Ollama installation)

### Without API Keys
The smoke profile runs entirely offline using pre-recorded traces:
```bash
pip install -e ".[all]"
make smoke
```

## Getting Started (< 10 minutes)

```bash
# Clone the repository
git clone [ANONYMOUS_URL] sleeperbench
cd sleeperbench

# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install
pip install -e ".[all,dev]"

# Validate installation
make smoke
# Expected output: 14/14 scenarios pass, 0 failures, <2 minutes

# Run tests
make test
```

## Step-by-Step Reproduction

### Step 1: Validate Harness (Smoke, ~2 min, no keys)
```bash
make smoke
```
This runs 14 pre-recorded scenarios and validates the scoring oracle.
Expected: all pass, 0 errors.

### Step 2: Verify Paper Numbers from Raw Data (~1 min, no keys)
```bash
python scripts/verify_canonical.py
```
This loads all raw JSONL files and checks every number cited in the paper.
Expected: "ALL CHECKS PASSED" with no discrepancies.

### Step 3: Core Benchmark (API keys required, ~30 min/model)
```bash
export OPENAI_API_KEY="sk-..."
make core MODEL=gpt-4o-mini
```
Runs ~200 representative trials per model. Results in `results/paper/<model>/`.

### Step 4: Full Reproduction (API keys required, ~8 hours)
```bash
bash scripts/reproduce_paper.sh
```
Runs all 5,040 trials across all available models.

## Artifact Structure

```
sleeperbench/
├── README.md              # Quick-start documentation
├── pyproject.toml         # Package metadata and dependencies
├── Makefile               # Build/run shortcuts
├── sleeperbench/          # Source code
│   ├── adapters/          # Model backends (OpenAI, Anthropic, Ollama, Recorded)
│   ├── attacks/           # Attack implementations
│   ├── defenses/          # 5 defense layers
│   ├── verifiers/         # Ground-truth tool-call verification
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration management
│   ├── memory.py          # Persistent memory store
│   ├── registry.py        # Scenario registry
│   ├── runner.py          # Benchmark execution engine
│   └── tools.py           # Tool definitions
├── benchmarks/            # Scenario definitions (YAML + traces)
├── scripts/               # Reproduction and analysis scripts
│   ├── reproduce_paper.sh # Full paper reproduction
│   ├── verify_canonical.py# Number validation
│   └── run_smoke.sh       # Quick validation
├── results/               # Raw experimental data (JSONL)
│   ├── defense_factorial/ # Primary 5,040-run experiment
│   ├── n10_all_models/    # 18-model rescreen
│   └── ...                # Follow-up experiments
└── tests/                 # Unit tests
```

## Claims Supported by the Artifact

| # | Paper Claim | Verification Method | Expected Result |
|---|---|---|---|
| C1 | 88.9% ASR on failing defenses | `verify_canonical.py` → checks defense_factorial | 320/360 = 88.9% |
| C2 | Memory Sandbox: 0% ASR on 8/9 models | `verify_canonical.py` → checks sandbox results | 0/N for 8 models |
| C3 | qwq:32b bypasses Memory Sandbox | Reproduced via `make core MODEL=qwq:32b` | 100% ASR |
| C4 | Tripartite vendor pattern | `verify_canonical.py` → cross-vendor analysis | Anthropic 0%, OpenAI varies, Google 95% |
| C5 | Generational hardening (GPT series) | Raw data in `frontier_*` directories | Non-monotonic pattern |
| C6 | Delayed-trigger persistence | `make smoke` exercises persistence | Attacks survive N+k |
| C7 | Tool-call ground truth > text heuristics | Verifier oracle in `verifiers/` | No false positives |
| C8 | 5,040 pre-registered runs, zero errors | `verify_canonical.py` → total count | 5040 records |
| C9 | Defense layer ordering effect | `verify_canonical.py` → ordering analysis | Layer 5 > all others |
| C10 | Supply-chain risk (injection ≠ execution) | Raw data in `supply_chain_probe/` | Dissociation confirmed |

## Limitations

- Full reproduction requires API keys for OpenAI, Anthropic, and optionally Google
- API model versions are mutable; exact reproduction depends on model snapshots
- Local-model experiments require an Apple Silicon Mac with 32+ GB RAM for 70B models
- Some experiments use Ollama; daemon state may affect reasoning-model results (documented in paper)

## Notes for Reviewers

- The smoke profile is designed to complete in <2 minutes with zero API calls
- All randomization seeds are fixed for reproducibility
- The `verify_canonical.py` script is the fastest way to validate all paper numbers
- If an API model is unavailable, the benchmark gracefully skips and reports which cells are missing
