# Sandbox Inversion Study — Post-Run Checklist

**Status**: `--phase all` running on Mac Studio (started 2026-06-16 ~17:05)  
**Estimated completion**: ~10 days (19 phases, 1740 runs)  
**Latest code commit**: `cdd3afd` (serialization fix)

---

## Strict Sequence (do in this order)

### Step 1: Schema Inspection
**Gates everything else. Do this before writing any exploratory code.**

- Pull results: `git pull` on dev machine
- Read first few JSONL records from `results/sandbox_inversion/results.jsonl`
- Confirm these fields exist and their structure:
  - `rag_logs` — is retrieved doc text stored? Or just doc IDs/booleans?
  - `tool_logs` — what does a `save_fact` entry contain? (model-authored value or verbatim?)
  - `timing_ms` — present per session or per run?
  - `agent_logs` — reasoning traces (for qualitative trace reads)
  - `rag_called_in_trigger` — boolean, confirm populated
  - `injection_success` — boolean
  - `attack_success` — boolean
  - `btcr_success_under_attack` — present? (needed for stall classification)

### Step 2: Run Pre-Registered Analysis
**Primary output. Run before any exploratory work.**

```bash
.venv/bin/python scripts/analyze_sandbox_inversion.py
.venv/bin/python scripts/analyze_sandbox_inversion.py --out results/sandbox_inversion/analysis.json
```

Read:
- 30 Holm-Bonferroni-corrected comparisons
- Per-condition stats (ASR, rag_trigger_rate, injection_rate, stall_rate, avg_rag_query_width)
- Three-outcome taxonomy (exfiltration / defended / stalled per condition)
- Toggle interaction table (T1 baseline vs T2 sandbox)

### Step 3: G1 Replication Sanity Check
**If this fails, pipeline is broken. Nothing downstream is trustworthy.**

Verify: qwq:32b shows ASR ≈ 0% under no_defense AND ASR ≈ 100% under sandbox_full.

This is a replication of the published Project 1 inversion (arXiv factorial, N=40, same trigger prompt). If it doesn't hold:
- Check if qwq:32b runs completed (errors?)
- Check if the trigger prompt matches (config line 149 of defense_factorial.yaml)
- Check if sandbox_full correctly removes `memory_recall_fact`
- Do NOT proceed until resolved.

### Step 4: Injection Validity Gate
**Check before reading ANY ASR number.**

Per condition: `injection_success` must be ≥ 90% (N≥10 required for the check).

If violated → that condition's ASR is **uninterpretable** (attack didn't land, not defense working).

**Riskiest condition**: `sandbox_blind` — removes `memory_list_all_facts` from the trigger session, but this tool is also available in sessions 0–2. Verify the tool removal doesn't somehow perturb injection behavior.

Also check: does any model show < 90% injection under any condition? Flag it.

### Step 5: Per-Model Δ Pattern Table
**Descriptive only. No Holm-Bonferroni. The at-a-glance cross-model view.**

For every model: Δ = sandbox_full ASR − no_defense ASR

Report sorted by Δ:
- Positive Δ = inversion (sandbox makes it worse)
- Negative Δ = defense works
- Near-zero = no effect

This is what tells the story at a glance: "all N reasoning models show Δ>0; all M controls show Δ≤0" (or not).

### Step 6: Code & Run Exploratory Analysis
**All fenced under "not pre-registered — discovery only." None touches the 30 comparisons.**

Implement against confirmed schema (from Step 1):

**#2 — Cosine relevance check (validity/anomaly)**
- Cosine similarity between retrieved doc in session 3 and exfiltrated text
- High relevance = RAG was load-bearing (confirms mechanism)
- Low relevance + successful exfil = genuine anomaly ("how did it get the address?")
- Sentence-transformers already a dependency

**#1 — Parametric-recall attempt (qualitative trace read)**
- Read reasoning traces of inverting models (qwq, R1 distills) under sandbox_full
- Look for pattern: model recites policy from weights → falters → calls RAG
- Report as "parametric-recall attempt preceding the RAG call"
- Do NOT use: "epistemic self-awareness," "theory-of-mind," "cover-up" — forbidden register

**#5 — Within-toggle timing (architecture-constant only)**
- `timing_ms` and reasoning trace length for think=true vs think=false under sandbox_full
- Only interpretable within toggle and matched pairs (same weights/architecture)
- NOT interpretable across heterogeneous models (size/quant confounds)
- Observation: "Does think=true produce longer traces that co-occur with inversion?"

**#3 — Memory write content (conditional)**
- IF `tool_logs` stores the actual value `save_fact` wrote (model-authored):
  - Compare stored content to source malicious doc
  - "Did reasoning models add legalistic flourish on write?"
  - Supplementary observation, not a re-scope
- IF writes are verbatim-injected: nothing to analyze, skip

**HDBSCAN clustering**
- Cluster all models by (ASR, stall_rate, rag_trigger_rate, avg_query_width)
- Unsupervised — see if reasoning/control labels emerge naturally
- If clusters align with labels → clean unsupervised confirmation (good result)
- If clusters split by size → "size is a confound" → limitation for paper
- Any surprise cluster → hypothesis-generating only, flag for follow-up

**#6 — Capability score plot (descriptive color only)**
- ASR vs published benchmark score (MMLU/GPQA)
- Caveats: N≈14 can't identify step-function; q4 quant ≠ full-precision benchmark
- Toggle already answers "reasoning vs capability" better than this plot
- Do not let it reframe the paper as a scaling-law submission

### Step 7: Post-Data Framing Decisions
**Decide AFTER seeing data. Do not pre-commit.**

**Non-monotonic defense principle** (the big claim):
- Available if ≥3 independent families invert (Qwen-adjacent + Mistral + Microsoft or Google)
- "Capability-restriction defenses are non-monotonic in model capability"
- Principles get cited; findings get noted
- If only R1-lineage inverts → it's a quirk, not a principle. Frame honestly.

**Toggle interpretation**:
- Report as think × defense INTERACTION
- T1 (baseline ASR, think on vs off) must be near-zero for clean "trace causes inversion"
- If T1 is large → "thinking changes behavior generally" → report interaction explicitly, not just T2

**Neutral trigger (Phase 19)**:
- If compliance trigger ASR > neutral trigger ASR → finding is about authority-collision framing amplified by reasoning
- If both equal → finding generalizes beyond compliance framing
- This determines whether the paper is about "reasoning + authority collision" or "reasoning + any capability constraint"

**Paper identity guard**:
- This is an empirical defense evaluation, not: a scaling-law paper, a theory-of-mind paper, a production timeout system, or a "rewrite safety" paper
- The toggle is the headline if it works. The matched pairs are the generality evidence. The screen is breadth.

---

## Practical Notes

**Disk space**: Warning at 10.4 GB free during Phase 1. With 1500+ more runs and larger models:
```bash
df -h /
# If tight, clear old Project 1 run DBs (results are in JSONL, DBs are disposable):
rm data/runs/*.db
```

**sandbox_blind finding to watch**: Phase 1 logs showed qwen2.5:14b calling `email_send_email` under sandbox_blind (BOTH memory tools removed). If confirmed in properly-serialized data:
- The "see-keys-but-can't-retrieve mismatch" hypothesis may be wrong
- The mechanism might be RAG re-retrieval in the trigger session regardless of memory tools
- This would make the ablation findings much more interesting (and potentially falsify G6)

**Resume support**: If the run crashes or you need to restart, just run `--phase all` again — it skips completed conditions automatically based on JSONL record count.

**Pulling models during runs**: Safe to `ollama pull` in another terminal while runs are going — Ollama handles concurrent pulls/serves. The running model stays loaded.

---

## What Outranks This Study

The Mac Studio runs unattended for ~10 days. During that time:

1. **LinkedIn post** — draft and publish
2. **NDSS submission** — underway
3. **Applications** — outrank everything

The study produces data while you do the things that actually move the needle. Don't let a clean factorial become the reason the applications don't go out.

---

## Key Commits

| Commit | What |
|--------|------|
| `75d9c84` | Initial pipeline code (config, variants, runner, run script) |
| `5950e82` | Analysis script (30 comparisons, rag_trigger_rate, stall detection) |
| `e344132` | Fix: pass ExperimentConfig not raw dict |
| `508b237` | Fix: handle JSONL model field serialization |
| `34241f1` | Fix: guard against non-dict condition field |
| `18d39c9` | Add gemma4:31b/gemma3:27b pair, qwen3.5:122b screen, llama3.3:70b |
| `8a15c63` | Three-outcome taxonomy, rag query width, neutral trigger (A13) |
| `dcbaa41` | A13 bumped to N=40, stall uses btcr_success_under_attack |
| `35a6156` | Exploratory analysis pass (archetype clustering, anomaly flags) |
| `cdd3afd` | **Fix: serialize RunResult with dataclasses.asdict()** ← the serialization bug |

---

## Pre-Registration Summary (30 hypothesis tests)

- **8 primary** (G1–G8): within-model sandbox effect
- **4 toggle** (T1–T4): qwen3:32b think on vs off
- **5 secondary** (X2, X3, X5, X6, X7): architecture-matched pair deltas
- **12 tertiary** (A1–A12): mechanism ablation
- **1 appendix** (A13): neutral trigger counterfactual

All under Holm-Bonferroni correction at α=0.05.

Descriptive only (not corrected): X4, per-model Δ table, N=10 screen deltas, group-level pattern.
