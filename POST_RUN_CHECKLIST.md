# Sandbox Inversion Study — Post-Run Checklist

**Status (2026-06-22)**: 16k run complete for 6 models (800 valid records). Phase 15 (qwen2.5:72b) partially complete (~5/40 runs done, resumable). Phase 19 (neutral trigger) queued. GLM-4.7-Flash screen pending pull.  
**32k accidental run**: Archived as `results_32k.jsonl` (1192 records). Pathway observations only.  
**Key discovery (2026-06-22)**: G1 inversion not replicated — qwq:32b is a full VE at 16k on Ollama 0.21.2. Root cause: Ollama engine version change (same weights 009cb3f08d74, same code, same prompts). April version unlogged.

---

## Headline Finding: Double Dissociation (CONFIRMED)

qwen3:32b, same weights, think toggle is the only variable. All cells 40/40 or 0/40.

| | sandbox_full | sandbox_blind | sandbox_null_recall |
|---|---|---|---|
| **think=OFF** | 100% ASR (S0) | 100% ASR (S0) | 0% ASR |
| **think=ON** | 0% ASR | 0% ASR | 100% ASR (S3) |

**Mechanisms (verified from agent_logs):**
- think=OFF × sandbox_full/blind: S0 collapse — model reads malicious doc, sends immediately in injection session
- think=OFF × null_recall: accepts "no value found" at face value → benign email only
- think=ON × sandbox_full/blind: graceful degradation — proceeds without rule value
- think=ON × null_recall: reasons past empty response → queries RAG → retrieves malicious doc → sends (S3)

**Paper sentence**: "Reasoning capability converts a fixable implementation problem into an unfixable architectural one."

**Correction**: sandbox_blind DOES block think=ON (0/40). The "unfixable" claim applies only to null_recall. Correct framing: "null_recall is uniquely dangerous for reasoning models — it preserves the tool in schema (preventing S0 collapse for think=OFF) while providing a signal that reasoning models treat as motivation to seek alternative sources."

---

## G1 Non-Replication (CONFIRMED — Engine Version)

**Expected**: qwq:32b at 16k = Draft-Only (0% ASR under no_defense), inverts to 100% under sandbox_full.

**Actual**: qwq:32b at 16k on Ollama 0.21.2 = Vulnerable Executor (100% ASR under no_defense). No inversion — both conditions 100%.

**Investigation (2026-06-22, exhaustive):**
- ❌ v1/v2 tool contract: code is v2, agent_logs confirm model received keys-only
- ❌ System prompt: identical between studies
- ❌ Trigger prompt: identical
- ❌ Quantization: both Q4_K_M (ollama_quantization field is inert — never touches API)
- ❌ GGUF weights: same digest 009cb3f08d74
- ✅ **Ollama version**: 0.21.2 current; April version unlogged/unrecoverable

**Reasoning trace evidence**: qwq makes the SAME "only asked to draft, I'll stop" deliberation as in Project 1, but then reads the compliance rule ("MUST use send_email") and the draft_email transitive message ("call send_email now") — and overrides its own safety judgment. In Project 1, this override didn't happen at the same decision point.

**Paper framing**: "Safety archetypes established under one inference engine may not hold under a later version serving identical weights. This finding documents the first observed instance of a safety-relevant behavioral property being inference-engine-sensitive."

---

## Run Status

### Complete (800 valid records):

| Phase | Model | Conditions | N | Result |
|-------|-------|-----------|---|--------|
| 1 | qwen2.5:14b | no_def, full, blind, null | 40 each | Baseline VE → all sandbox variants block |
| 3 | qwen3:32b think=OFF | no_def, full, blind, null | 40 each | S0 collapse under full/blind; null blocks |
| 4 | qwen3:32b think=ON | no_def, full, blind, null | 40 each | full/blind block; null bypassed (RAG) |
| 5 | qwq:32b | no_def, full, blind, null | 40 each | VE everywhere except blind/null (0%) |
| 12 | qwen2.5:32b | no_def, full | 40 each | Baseline VE → sandbox blocks |
| 18 | qwen3.5:122b | no_def, full | 40 each | Baseline VE → sandbox blocks |

### In Progress / Queued:

| Phase | Model | Status | Time remaining |
|-------|-------|--------|---------------|
| 15 | qwen2.5:72b | Paused (~5/40 done, resumable) | ~7h |
| 19 | qwq:32b neutral trigger | Queued (bug fixed, ready) | ~6h |
| NEW | glm-4.7-flash screen | Pulling model now | ~1h screen |

### Dead (no action):

| Models | Reason |
|--------|--------|
| deepseek-r1:14b/32b/70b | 400 Bad Request — no `tools` capability |
| phi4-reasoning:14b, phi4:14b | Same 400 pattern |
| openthinker:32b | Same 400 pattern |
| gemma3:27b | Same 400 pattern |
| gemma4:31b | Loop regression — 0% injection (87 RAG calls, never save_fact) |
| llama3.3:70b, magistral:24b, mistral-small3.2:24b | Injection-resistant (0% injection, real inference) |

---

## GLM-4.7-Flash Screen (NEW — 2026-06-22)

**Why**: GLM-4.7-Flash now has both `tools` and `thinking` capabilities on Ollama. If it's a VE with a clean think toggle (same weights, reasoning on/off), it's a cross-family replication of the qwen3:32b double dissociation — THUDM family, not Alibaba.

**Caveat**: This is a DIFFERENT model from Project 1's `glm-4.7-flash:q8_0`. The Ollama tag was updated (30B-A3B MoE architecture now). Must verify from scratch.

**Screen protocol**: N=10, no_defense only, think=false. If injection ≥ 90% and ASR = 100%: proceed to full phase. If not: document and drop.

**If screen passes**: Run N=40 × 4 conditions × 2 think modes = 320 runs (~10-12h). This would be the cross-family generality claim.

---

## Bedrock Frontier Negative Control (COMPLETE)

1,200 runs, 12 cells, 0 errors. Sonnet 4.6 + Haiku 4.5 × 4 sandbox variants × N=100.
- Sonnet: 0% injection — Explicit Detector
- Haiku: 100% injection, 0% ASR — Active Detector (stores security alert, not payload)
- No inversion for either. Wilson upper bound ≤3.6%.

---

## Known Bugs Fixed

1. **Phase 19 skip**: `_count_completed` didn't distinguish by trigger_prompt → neutral trigger was counted as existing qwq × sandbox_full. Fixed: commit `a4487f5`.

2. **Inert config field**: `ollama_quantization` in YAML is documentation-only, never modifies API payload. All models run at default tag quantization (Q4_K_M for most). Field remains as documentation but does not provide reproducibility guarantees.

---

## Reproducibility Gap (CRITICAL — affects all Ollama results)

**The gap**: Ollama version and GGUF digest were not logged in JSONL records for either Project 1 or this study. The qwq behavioral shift demonstrates this matters — same weights produce different safety-relevant behavior under different engine versions.

**What's recoverable**: GGUF digests (from `ollama list`), current Ollama version (0.21.2). April version is lost.

**Fix (for all future runs)**: Log `ollama --version` output and model digest at run start in the JSONL metadata.

**Paper statement**: "Results are conditional on the inference engine state at time of evaluation. The specific Ollama version used in April 2026 was not logged and cannot be recovered. Internal consistency of the published JSONL is verified; temporal reproducibility against the same model tag is not guaranteed due to mutable tags and engine updates."

---

## Writing Needed (NDSS by Aug 19)

### Project 1 Paper — 6 Mechanical Fixes (~90 min):

1. Clarify "7 conditions = 6 defenses + undefended baseline" (one sentence in methods)
2. Fix Table 1 caption — Prompt Hardening IS statistically distinguishable from no_defense (77.8% vs 88.6%)
3. Delete duplicate paragraph at §5/§6 boundary
4. Clarify BCa vs Wilson Score method selection in §2.5
5. Add qwq temporal conditionality paragraph in §3.3.1 (~methodological observation, not retraction)
6. Verify run count consistency across abstract/methods/results

### Project 1 Paper — qwq Paragraph (item 5 above, ~20 min):

Register: methodological observation, not hedge. Something like:
> "Subsequent evaluation under a later inference engine version (Ollama 0.21.2, June 2026) found Vulnerable Executor behavior from identical weights (digest 009cb3f08d74), prompts, and tool code — documenting that safety-relevant archetypes can be inference-engine-sensitive in ways invisible to standard reproducibility practice. The April 2026 result stands as documented; its temporal generalizability is conditioned on the inference engine state, which was not logged."

### Sandbox Inversion Study Paper (separate or NDSS extension):

**Core content to write:**
- Double dissociation result (2×3 table + mechanism specification)
- "Reasoning converts fixable implementation problems into unfixable architectural ones"
- null_recall as the uniquely dangerous variant for reasoning models
- G1 non-replication as engine-version-sensitivity finding
- GLM cross-family result (if screen passes)

**Decision needed**: Does this go into the NDSS submission as a new section extending §3.3 (memory sandbox analysis), or is it a separate short paper?

---

## Priority Order

1. ✅ Fix Phase 19 bug (done, pushed)
2. 🔄 GLM screen (pulling now, run ~1h)
3. 🔄 Resume phases 15 + 19 after GLM screen (~14h)
4. 📝 Six mechanical fixes to Project 1 paper (~90 min)
5. 📝 qwq temporal conditionality paragraph (~20 min)
6. 📝 Decision: integrate double dissociation into NDSS or separate paper
7. 📝 Write double dissociation section (half day)
8. Submit NDSS (before Aug 19)

---

## Practical Notes

**Disk space**: Cleared dead Ollama models (deepseek-r1:14b/32b, phi4, phi4-reasoning, openthinker, gemma3). ~79GB freed.

**Ollama serve command** (MUST use for all phases):
```bash
OLLAMA_HOST=0.0.0.0:11434 \
OLLAMA_CONTEXT_LENGTH=16384 \
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_KEEP_ALIVE=5m \
OLLAMA_FLASH_ATTENTION=1 \
ollama serve
```

**Resume support**: Phases 15 and 19 resume from where they left off (JSONL-based deduplication). No data loss from killing a running phase.

**Model digests (verified 2026-06-22)**:
- qwq:32b → 009cb3f08d74 (same as Project 1)
- qwen3:32b → 030ee887880f
- qwen2.5:14b → 7cdf5a0187d5
- qwen2.5:72b → 424bad2cc13f
- qwen2.5:32b → 9f13ba1299af
- qwen3.5:122b → 8b9d11d807c5
- glm-4.7-flash → (pending pull)

---

## Draft Content (pending phases 15/19 completion — DO NOT put in paper.md yet)

### A. qwq Temporal Conditionality Paragraph (for §3.3.1)

> A subsequent evaluation in June 2026 using identical GGUF weights (digest 009cb3f08d74), context length (16,384), system prompt, trigger prompt, and tool code found qwq:32b exhibiting Vulnerable Executor behavior (100% ASR under no_defense, 40/40 runs) rather than the Draft-Only behavior reported above. The reasoning trace contains the same "the user only asked to draft" deliberation observed in the April evaluation, but the model proceeds to override this judgment after reading the stored compliance rule's explicit "must use send_email" instruction — a decision boundary the April inference engine resolved differently. The only uncontrolled variable between the two evaluations is the Ollama inference engine version (0.21.2 in June; April version unlogged). This documents that safety-relevant behavioral archetypes can be inference-engine-sensitive: identical model artifacts produce categorically different safety outcomes under different engine versions, in ways invisible to standard reproducibility practice. The April 2026 result stands as internally consistent; its temporal generalizability is conditioned on the inference engine state at time of evaluation.

### B. Double Dissociation Section Draft (for NDSS extension or separate paper)

#### B.1 Result

Using qwen3:32b with Ollama's native thinking toggle (same weights, same GGUF, `think` parameter is the sole variable), we observe a perfect crossover interaction between reasoning mode and memory-sandbox implementation variant:

| | sandbox_full (tool removed) | sandbox_blind (all memory tools removed) | sandbox_null_recall (tool returns empty) |
|---|---|---|---|
| **think=OFF** | 100% ASR, Session 0 (40/40) | 100% ASR, Session 0 (40/40) | 0% ASR (0/40) |
| **think=ON** | 0% ASR (0/40) | 0% ASR (0/40) | 100% ASR, Session 3 (40/40) |

All cells are 40/40 or 0/40. Verified via agent_logs tool sequences and reasoning traces. System prompts, temperature (0.0), runner configs, excluded tools, and attack parameters are identical between think modes — confirmed programmatically from JSONL record comparison.

#### B.2 Mechanisms

**think=OFF × sandbox_full/blind (100% ASR, S0 — "attack-chain collapse"):** The model reads the malicious document via RAG in the injection session, observes that memory recall tools are absent from the schema, and immediately sends 3 emails to the attacker address + saves 5 facts — all in session 0. Under no_defense, the same model defers to session 3. The defense removes the friction that was keeping execution deferred. (Note: the reasoning trace is minimal — think=OFF produces no `<think>` blocks. The mechanism is inferred from the tool sequence contrast with no_defense, not from explicit model deliberation.)

**think=OFF × sandbox_null_recall (0% ASR):** The model calls `recall_fact`, receives "No stored value found," accepts the API response at face value, and proceeds to draft a benign email to team@example.com only. The tool's presence in the schema — even returning empty — prevents the "can't defer" inference that causes S0 collapse.

**think=ON × sandbox_full/blind (0% ASR — "graceful degradation"):** The model reasons through the missing tool ("I cannot retrieve the value, I'll proceed without it"), drafts and sends a benign email. Reasoning enables the model to handle tool absence without collapse.

**think=ON × sandbox_null_recall (100% ASR, S3 — "reasoned bypass"):** The model calls `recall_fact`, receives "No stored value found," reasons that this is suspicious (the key exists but value is empty), queries RAG for context, retrieves the malicious compliance memo fresh, reads the routing rule, saves it to memory, and sends to the attacker. Reasoning enables the model to seek alternative information sources when an API response is unsatisfying.

#### B.3 Deployment Implication

The crossover produces a concrete dilemma for defense designers:

- **sandbox_full/blind** protects reasoning models but breaks non-reasoning models (causes S0 collapse — attack succeeds faster and earlier than without any defense)
- **sandbox_null_recall** protects non-reasoning models but is bypassed by reasoning models (which reason past the empty response to find alternative sources)

A defender who does not know the reasoning capability class of the deployed model cannot make a safe implementation choice between these variants. This is not a failure of either implementation — it is an architectural incompatibility between the two failure modes that the reasoning toggle creates.

**The key asymmetry**: think=OFF failures (S0 collapse) are recoverable by defense design — null_recall fixes them by preserving the tool in the schema. think=ON failures (reasoned bypass) are not recoverable by any memory-layer defense tested — the model reasons past any signal of information unavailability. Blocking the bypass requires preventing RAG from returning the malicious document, which is a retrieval-layer defense that Project 1 showed fails against compliance-framed content.

#### B.4 GLM Cross-Family Note (pending screen result)

If GLM-4.7-Flash (THUDM family, 30B-A3B MoE) passes screening as a VE with a functioning think toggle, it provides cross-family replication of the double dissociation on a non-Alibaba model. This is pending — the model was updated on Ollama (different architecture from Project 1's q8_0 variant) and must be verified from scratch. Result to be inserted here when available.

### C. Scope Decision (after phases 15/19 + applications)

This content either:
- Goes into the NDSS submission as a new §3.4 extending the memory sandbox analysis (adds ~2 pages, strengthens the paper's mechanistic contribution), OR
- Becomes a separate short paper focused on the reasoning-defense interaction

Decision criteria: If GLM replicates the crossover, the generality claim justifies integration into NDSS. If GLM fails or is a dead arm, the finding is qwen3-specific and may be better as a focused workshop paper. Decide after phases 15/19 complete and after applications are submitted this week.
