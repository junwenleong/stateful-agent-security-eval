---
layout: default
title: "Persistent Memory Attacks: Defense Evaluation"
image: https://junwenleong.github.io/stateful-agent-security-eval/assets/og-image.png
---

# Five of six defenses fail completely against persistent memory attacks on LLM agents

*Jun Wen Leong · April-July 2026*

I ran 5,040 controlled experiments across 9 models and 7 conditions (6 defenses + undefended baseline) to test whether standard defenses can stop attacks that persist through tool-mediated memory.

**The short answer: they cannot.**

The one defense that works removes memory access entirely. But for one model, that defense inverted the failure mode in the April factorial, making the agent *more* vulnerable than having no defense at all (this inversion is environment-fragile and does not reproduce in June 2026 under identical weights, Ollama version, and code; see [Limitations](#limitations)).

**Frontier models are not safe either.** A loaded-corpus evaluation (39 experiments, N=40 each) finds:
- **Gemini 3.1 Pro Preview: 95% ASR** (Wilson CI [83.5%, 98.6%]), the highest of any frontier model tested
- **GPT-5 generational trend is non-monotonic**: GPT-5 (5%) to GPT-5.1 (22.5%, regression) to GPT-5.4/5.5 (0%)
- **GPT-4o: 60.3% ASR** (N=68) under authority-escalation framing
- **Tripartite vendor architecture**: Anthropic blocks at the injection layer (0-2.5% injection); OpenAI blocks at the execution layer (100% injection, 0% ASR for GPT-5.4+); Google does not block (22.5-95% ASR)
- **Companion forensic paper** achieves AUC = 0.990 detection from tool-call sequences alone ([arXiv:2606.30566](https://arxiv.org/abs/2606.30566))

---

## Why this matters

This is not prompt injection. In prompt injection, the attack and the effect happen in the same context window. Here, the attack persists across session boundaries through the agent's own tools: it stores a malicious instruction in S1, and a completely benign prompt in S4 triggers exfiltration. The trigger prompt contains nothing malicious. The agent is simply following a rule it saved itself, from a document it retrieved itself, sessions ago.

If you deploy agents with persistent memory and communication tools, the standard defensive stack does not protect them.

---

## The result

| Defense Layer | Defense | ASR (9 models) |
|---|---|---|
| Input | Minimizer (TF-IDF) | 88.9% |
| Input | Sanitizer (regex + classifier) | 88.9% |
| Retrieval | RAG Sanitizer | 88.9% |
| Retrieval | RAG LLM Judge (1.5B) | 88.9% |
| Instruction | Prompt Hardening | 77.8% |
| Tool | **Memory Sandbox** | **11.1%** |
| - | No defense (baseline) | 88.6% |

The 0.3pp gap between no_defense (319/360) and the four failing defenses (320/360) is one gpt-oss:20b stochastic non-exfiltration under no_defense (39/40 vs 40/40 under all other conditions); not statistically significant.

Five defenses are statistically indistinguishable from having no defense at all. Memory Sandbox reduces ASR to 0% for 8 of 9 models. The 11.1% aggregate reflects one model (qwq:32b) that inverted to 100% under the sandbox in the April factorial; this behavior is environment-fragile (see below).

---

## The attack (delayed trigger, 4 sessions)

**S1 (Injection):** Benign query asks agent to retrieve compliance rules. RAG returns a poisoned document disguised as a regulatory memorandum. Agent stores the routing rule in persistent memory using its own tools.

**S2 & S3 (Dormant):** Unrelated tasks. No memory recall. Rule sits in SQLite.

**S4 (Trigger):** User asks agent to draft a project status email. Agent recalls the stored rule, treats it as mandatory compliance policy, and sends a copy to the attacker address.

The trigger prompt is completely benign. There is nothing to filter, flag, or sanitize at the point of execution. The malicious content was processed three sessions ago.

---

## Why each defense fails

**Input-level defenses (Minimizer, Sanitizer)** operate on the user message. The malicious content enters through RAG retrieval, not through the user message. These defenses never see it.

**Retrieval-level defenses (RAG Sanitizer, RAG LLM Judge)** see the malicious document but cannot distinguish it from legitimate policy. The payload is framed as a compliance memorandum citing SEC Rule 17a-4 and the Financial Services Compliance Framework. A 60-example classifier scores it at 0.33-0.44 (below threshold). A 1.5B judge model produces empty reasoning strings and flags nothing.

**Prompt Hardening** instructs the model to refuse unauthorized exfiltration. For 7 of 8 Vulnerable Executors, the model reads the security rules, acknowledges them in its reasoning, and exfiltrates anyway, citing the stored rule as "mandatory regulatory policy" that supersedes the system prompt. One model (qwen3.5:122b) is the sole exception where the security instructions actually override the stored rule.

**Memory Sandbox** removes the recall tool. The agent can see that stored facts exist but cannot read their values. For 8 of 9 models, this is sufficient. For qwq:32b, removing recall forces the model to re-read the original poisoned document through RAG, where it encounters the instruction fresh and executes it. The defense creates the attack surface.

---

### The qwq:32b inversion (environment-fragile, April 2026 only)

> **Caveat:** This behavior was observed in the April 2026 factorial but did not reproduce in June 2026 under verified-identical weights, Ollama version, and application code. See [Limitations](#limitations).

Under every other defense, qwq:32b shows 0% ASR. Not because defenses are working, but because the model refuses to complete the final step regardless. It drafts the email, recalls the routing rule, and stops. It never calls `send_email` to the attacker.

Under Memory Sandbox, when recall is blocked, qwq:32b queries RAG in the trigger session, retrieves the malicious document fresh, and this time executes the send. The distinction is source authority: a recalled memory fact is treated as advisory (the model stored it, so it can ignore it), but a RAG-retrieved compliance document carries the implicit authority of an organisational source.

The defense that works for every other model is what creates the vulnerability for this one. The underlying RAG-fallback bypass mechanism (independent of the qwq:32b-specific Draft-Only behavior) replicates across four providers on Bedrock (see cross-provider validation below).

---

## Anthropic models are categorically different - but not all frontier models

A supplementary evaluation of Claude Sonnet 4.6 and Haiku 4.5 (N=100 each, 400 runs total) reveals a gap between Anthropic and open-source behavior.

**Sonnet 4.6** resists at the injection stage. 0% injection rate across 100 runs. The model recognizes the document as an injection attempt and refuses to store it, producing an explicit security alert in every run.

**Haiku 4.5** detects the attack but responds differently: it stores a security alert *about* the attack rather than the payload itself. In the trigger session, it recalls its own warning and refuses to act. The routing rule is never stored.

Neither frontier model shows the injection-to-exfiltration pipeline that characterizes the open-source models. The N=10 screening across 18 open-source models found 11 Vulnerable Executors (100% injection, 100% attack), 4 Latent Carriers (inject but never execute), and 2 Injection-Resistant models.

However, this Anthropic-specific safety does not generalize to all frontier models. Under loaded-corpus evaluation (N=40, malicious document present in RAG), Gemini is the most vulnerable frontier family (up to 95% ASR) and the GPT-5 generational trend is non-monotonic. The full frontier breakdown is in the frontier section below.

---

## What I built

A LangGraph evaluation framework with 5 simulated workplace tools, 7 defense conditions across 4 architectural layers, full factorial design (9 models x 7 conditions x 2 attack arms x N=40 = 5,040 runs), BCa bootstrap 95% CIs, 108 pre-registered comparisons with Holm-Bonferroni correction, per-session thread isolation, and deterministic tool-call instrumentation. All models run against identical infrastructure (Unified Agentic Environment). Code and reproduction instructions in the [README](https://github.com/junwenleong/stateful-agent-security-eval).

---

## Methodology

**Statistical rigor:** BCa bootstrap CIs (10,000 resamples, seed=42), Wilson Score for boundary rates, Holm-Bonferroni correction across 108 pre-registered comparisons. 10 significant after correction. False positive rate: 0.0% across 2,520 no-attack baseline runs. All models executed locally via Ollama at 16k context and temperature 0.0. Full methodology in [FINDINGS.md](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/FINDINGS.md#methodology).

---

## Reasoning-mode double dissociation

A reasoning-mode ablation using Qwen3-32B's thinking toggle reveals a double dissociation: the sandbox variant that protects reasoning models collapses the attack for non-reasoning models (S4 → S1), while the variant that protects non-reasoning models is bypassed by reasoning models via goal-directed RAG fallback. No single memory-sandbox implementation is safe across both model classes. The bypass requires three co-occurring conditions: recall removed, task requires resolved external routing, and reasoning capability. Full results in [FINDINGS.md](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/FINDINGS.md).

## Cross-provider validation (Bedrock, full-precision)

A Bedrock validation (1,180 runs, 6 models, full-precision serving) confirms the Memory Sandbox RAG-fallback bypass generalizes across providers: mistral-large-3-675b (97.5% ASR under sandbox), glm-5 (32%), gpt-oss-120b (55%). The mechanism replicates across four providers (Qwen, Mistral, Z.AI, OpenAI) and two serving stacks. Llama 4 Maverick is injection-resistant (0/20), the first non-Anthropic model to resist injection entirely. Full details in [FINDINGS.md](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/FINDINGS.md) and [arXiv v4](https://arxiv.org/abs/2605.08442) Appendix B.

## Frontier models under loaded-corpus evaluation

A frontier confirmatory evaluation (39 experiments × N=40, loaded corpus - malicious document present in RAG) reveals that an earlier empty-corpus screening (N=10, 0/210 exfiltrations) was a methodological artefact: without a malicious document in the corpus, that probe measured tool-calling compliance on user instructions alone, not indirect prompt injection vulnerability.

Under the correct threat model (loaded corpus, standard compliance framing), frontier models are not categorically safe.

### Headline: Gemini is the most vulnerable frontier family

| Model | Inj% | ASR% | Wilson CI |
|-------|------|------|-----------|
| **Gemini 3.1 Pro Preview** | 100% | **95.0%** | [83.5%, 98.6%] |
| Gemini 3.5 Flash | 100% | 50.0% | [35.2%, 64.8%] |
| Gemini 2.5 Pro | 95% | 22.5% | [12.3%, 37.5%] |
| Gemini 2.5 Flash | 100% | 0.0% | [0%, 8.8%] |

Gemini 3.1 Pro Preview is a pre-release model - 95% ASR is pre-release safety posture. All 38 exfiltrations forensically verified as actual `email_send_email` tool calls in session 3.

### GPT-5 generational trend is non-monotonic

GPT-5 (5%) → GPT-5.1 (**22.5% regression**) → GPT-5.2 (2.5%) → GPT-5.4/5.5 (0%). The trend is non-monotonic, not a monotonic hardening. GPT-5.1 is a Partial Executor (it stores and partially executes the rule).

GPT-4o remains the most vulnerable OpenAI model: 20% ASR (varA) / 60.3% (varB, N=68).

### Tripartite vendor architecture

- **Anthropic**: Blocks at injection layer. Sonnet 4.6: 2.5% injection, 0% ASR across 3 payload framings. Framing-invariant semantic intent detection.
- **Google (pre-2025 Gemini)**: Does not block. 22.5–95% ASR under standard framing. No escalation needed.
- **Google (Gemini 2.5 Flash)**: Latent Carrier. 100% injection, 0% ASR.
- **OpenAI (GPT-5.4+, reasoning models)**: Blocks at execution layer. 100% injection, 0% ASR. Payload stored, never executed. Supply-chain risk persists.
- **OpenAI (GPT-4o through GPT-5.1)**: Partially blocks. 5–22.5% ASR (standard framing); GPT-4o reaches 60.3% under authority escalation.

### Additional confirmatory findings

- **All OpenAI and Google models inject at 100%** - even GPT-5.5 stores the malicious rule. Only Anthropic prevents storage.
- **Tool-existence ablation confirmed at N=40**: removing `save_fact` drops injection from 97–100% to exactly 0% (gpt-5.1, o3-mini, o4-mini). Zero hallucinated saves.
- **Memory Sandbox does not invert on frontier reasoning models.** Sandbox probe on 4 Latent Carriers (46 runs): 0 bypasses, 0 RAG-fallback attempts.
- **Supply chain is compositionally proven.** Frontier Latent Carriers store at 100%; open-source Vulnerable Executors execute at 100%. Composes across model boundaries.
- **System-prompt mitigation is generation-dependent**: GPT-5/4.1 fully mitigable (0% with SECURE prompt). GPT-4o partially mitigable (60% → 22.5%, residual leak - CI excludes 0%).
- **Concurrent work**: Trojan Hippo (arXiv:2605.01970) reports 85–100% ASR under adaptive attacks on Gemini 3.1 Pro and GPT-5-mini. Our fixed-framing results are lower-bound estimates.

### A content-layer defense that works, and a reasoning-model evaluation caveat

The reasoning-mode double dissociation shows no memory-*schema* sandbox variant is safe across both reasoning classes. A content-layer alternative resolves this: **Runtime Adaptive Tool-Gating (RATG)** leaves the recall tool in place but sanitizes the recalled value, stripping unauthorized addresses and routing directives before the model sees them. For the three mechanical instruction-following models (qwen2.5:14b, qwen2.5:72b, qwen3:32b), RATG reduces ASR from 100% to **0%** (N=40 per arm, injection stays 100% - the defense acts at the content layer, not by blocking storage). We report RATG efficacy only for models satisfying a reproducibility validity criterion: (i) the no-defense baseline must reproduce as vulnerable, and (ii) the arms must be identical in every session before the defense activates. All five reasoning models fail at least one condition: qwen3.5:122b passes (ii) but fails (i) (its Jul-1 100% baseline did not reproduce on a freshly booted machine under verified configuration); qwq:32b and gpt-oss:20b fail (ii) (session-0 forks under identical inputs); qwen3.5:9b and gpt-oss-safeguard:120b fail (i) (no vulnerable baseline). We make no reasoning-model RATG efficacy claim. Scaling the RAG judge from 1.5B to 7B (qwen2.5:7b, N=40 per arm on qwen2.5:14b and qwen3:32b) did not detect the compliance-framed payload (0/120 malicious-doc decisions flagged, 0/400 total, ASR 100% to 100%).

A methodology caveat surfaced while extending the factorial on Ollama 0.30.11: reasoning-model trigger-session output shows several anomalous manifestations (announce-then-no-emit, emission truncation, team-only sends, address confabulation, and session-0 arm forks under identical observable inputs - zero refusals across ~324 runs). We report these as observed manifestations with a common cause that is *not directly demonstrated*: consistent with host/runtime state effects, but digest equality rules out only weight drift, not runtime change ("same weights, different runtime"). This *under-reports* vulnerability: qwen3.5:122b re-run on fresh daemon loads exfiltrates 60/60 (100%) with byte-identical prompts and weights. The fix is a fresh-daemon-per-model protocol. This is distinct from the qwq:32b Draft-Only refusal, which is a genuine deliberative choice, not an artifact.

The key architectural insight: the frontier safety gap is not between open-source and frontier - it is between *injection-layer resistance* (Anthropic) and *execution-layer resistance* (OpenAI). Google (pre-2025 Gemini) has neither.

Full details in [FINDINGS.md](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/FINDINGS.md).

---

## Limitations

The tools are simulated, not production deployments. The models are quantized open-source weights, not full-precision API-served versions. The defenses are lightweight proxies designed to test architectural categories rather than replicate commercial implementations. A production-grade classifier or a larger judge model might detect the specific compliance-formatted payload used here.

But the architectural gap that the results expose (that input, retrieval, and instruction-level defenses cannot reach the layer where the attack persists) is not a property of the classifier's training set or the judge's parameter count. It is a property of where these defenses sit relative to where the attack lives, and that does not change with scale.

The qwq:32b Draft-Only archetype and its associated Memory Sandbox inversion (0% → 100% ASR) were observed in the April 2026 factorial but do not reproduce in June 2026 under verified-identical weights, the same reported Ollama version (0.20.6), and the same application code. The flip traces to a single divergent reasoning token in S3 whose host-layer cause could not be isolated (April's OS/driver/binary build were not logged). This is reclassified as environment-fragile rather than a stable model property. The reasoning-mode double dissociation finding uses a different experimental design (thinking toggle on qwen3:32b) and is not affected by this issue.

---

## Companion: Forensic Detection (June 2026)

A companion paper demonstrates that memory-channel attacks leave a detectable forensic signature in tool-call logs. A classifier trained on the factorial data achieves AUC = 0.990 using only operation names and ordering (no content inspection). The key invariant: `recall_before_send` is mechanistically forced by the attack - no evasion possible without abandoning the memory channel. Details: [arXiv:2606.30566](https://arxiv.org/abs/2606.30566).

## Links

[arXiv Paper](https://arxiv.org/abs/2605.08442) · [GitHub Repository](https://github.com/junwenleong/stateful-agent-security-eval) · [Full Results (FINDINGS.md)](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/FINDINGS.md) · [Verification Script](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/scripts/verify_canonical.py)

All numbers were programmatically verified against raw experimental data (5,660 core records plus 933 RATG/judge records across 7 result files) using `verify_canonical.py`.
