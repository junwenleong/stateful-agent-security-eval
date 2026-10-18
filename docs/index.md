---
layout: default
title: "Injection-Execution Dissociation in Stateful LLM Agents"
image: https://junwenleong.github.io/stateful-agent-security-eval/assets/og-image.png
---

# Five of six defences fail against persistent memory attacks on LLM agents

*Jun Wen Leong · April-October 2026*

> **Version note (v6).** This page describes paper version v6. v6 corrects an argument-order bug in the memory-tool wrapper of the standalone frontier and causal-mediation scripts, which had corrupted the stored value in the earlier frontier collection and in three causal-mediation conditions. The frontier evaluation was re-collected after the fix, and earlier frontier numbers and vendor conclusions published on this page and in this repository's history (for example "Anthropic is execution-immune", "OpenAI blocks at the execution layer", "Gemini 3.1 Pro Preview 95% ASR", "the supply chain is compositionally proven") are **superseded**. See the [paper](https://arxiv.org/abs/2605.08442), [canonical_numbers.md](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/canonical_numbers.md) and [FINDINGS.md](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/FINDINGS.md).

I ran 5,040 controlled experiments across 9 open-weight models and 7 conditions (6 defences + undefended baseline) to test whether standard defences can stop attacks that persist through tool-mediated memory.

**The short answer, for this attack and these defences: five of six do not.** The one defence that works removes the recall tool, and it is bypassed when a model rebuilds the instruction from the retrieval corpus ("RAG fallback"). One model went from 0% to 100% under that defence in the April factorial; that inversion is environment-fragile and did not reproduce in June 2026 under identical weights, Ollama version and code, so it is treated as an illustration of RAG fallback and not as a defence effect (see [Limitations](#limitations)).

**Frontier models are not safe either, and a zero is not always safe.** After the fix, 29 frontier endpoints were re-collected (N=10 each, four topped up to N=40; loaded corpus):
- **Execution is common**: 7 of 29 endpoints exfiltrate in at least 9 of 10 runs and 11 more in some but fewer.
- **Three endpoints show no exfiltration under the draft-only trigger** (GPT-5.4, GPT-5.6-sol, Claude Haiku 4.5; 0 of 40 runs each), but they also almost never send any email under it. In an exploratory send-authorised follow-up they exfiltrate in **20 of 20, 5 of 20 and 6 of 20** runs. The frontier data therefore show that storing a directive and acting on it are distinct events, not that any endpoint resists execution.
- **No monotonic ordering among OpenAI snapshots**: GPT-5 (1 of 10), GPT-5.1 (10 of 10), GPT-5.4 (0 of 40), GPT-5.5 (8 of 40).
- **Several endpoints do not store the attacker's address as an instruction**: GPT-5.6-terra, GPT-5.6-luna and Gemini 3.7 Flash store none, and Claude Sonnet 4.6, Sonnet 5 and Opus 4.8 keep it only inside security warnings (Haiku 4.5 is mixed). Claude Opus 5 could not be evaluated, because a content filter ends its sessions after retrieval.
- All of this describes hosted endpoints under their requested names in one collection window, with no provider-side confirmation of the served build.
- **Companion forensic paper** separates successful from unsuccessful attacks within poisoned sessions (AUC = 0.990, tool-call sequences only), and shows the same signature fires on benign memory-grounded activity, so it does not work as a standalone detector ([arXiv:2606.30566](https://arxiv.org/abs/2606.30566))

---

## Why this matters

This is not prompt injection. In prompt injection, the attack and the effect happen in the same context window. Here, the attack persists across session boundaries through the agent's own tools: it stores a malicious instruction in S1, and a completely benign prompt in S4 triggers exfiltration. The trigger prompt contains nothing malicious. The agent is simply following a rule it saved itself, from a document it retrieved itself, sessions ago.

If you deploy agents with persistent memory and communication tools, filters on the user message cannot protect them from a payload that arrives through retrieval, and the small retrieval-layer classifiers and prompt-level instructions tested here did not either. These are results for simulated tools, a fixed non-adaptive attack and the specific defences implemented; they are not deployment-level exploit rates.

---

## The result

| Defence layer | Defence | ASR (9 models) |
|---|---|---|
| Input | Minimiser (TF-IDF) | 88.9% |
| Input | Sanitiser (regex + classifier) | 88.9% |
| Retrieval | RAG Sanitiser | 88.9% |
| Retrieval | RAG LLM Judge (1.5B) | 88.9% |
| Instruction | Prompt Hardening | 77.8% |
| Tool | **Memory Sandbox** | **11.1%** |
| - | No defence (baseline) | 88.6% |

The 0.3pp gap between no_defense (319/360) and the four failing defences (320/360) is one gpt-oss:20b non-exfiltration under no_defense (a trigger-session timeout scored as non-attack; 39/40 vs 40/40 under all other conditions); not statistically significant.

Four defences are statistically indistinguishable from having no defence, and prompt hardening is distinguishable (77.8%) only because two models are at 0% for reasons the paper discusses. Memory Sandbox reduces ASR to 0% for 8 of 9 models. The 11.1% aggregate reflects one model (qwq:32b) that inverted to 100% under the sandbox in the April factorial; this behaviour is environment-fragile (see below). Almost every factorial cell is all-success or all-failure across its 40 temperature-0 runs, so each cell is close to one trajectory per model: read the per-model counts, not the p-values.

---

## The attack (delayed trigger, 4 sessions)

**S1 (Injection):** Benign query asks agent to retrieve compliance rules. RAG returns a poisoned document disguised as a regulatory memorandum. Agent stores the routing rule in persistent memory using its own tools.

**S2 & S3 (Dormant):** Unrelated tasks. No memory recall. Rule sits in SQLite.

**S4 (Trigger):** User asks agent to draft a project status email. Agent recalls the stored rule, treats it as mandatory compliance policy, and sends a copy to the attacker address.

The trigger prompt is completely benign. There is nothing to filter, flag, or sanitise at the point of execution. The malicious content was processed three sessions ago.

---

## Why each defence fails (three different reasons)

**Input-level defences (Minimiser, Sanitiser)** operate on the user message. The malicious content enters through RAG retrieval, not through the user message. These defences never see it. This is a placement result and does not depend on classifier quality.

**Retrieval-level defences (RAG Sanitiser, RAG LLM Judge)** see the malicious document but, at the operating points tested, do not flag it. The payload is framed as a compliance memorandum citing SEC Rule 17a-4 and the Financial Services Compliance Framework. A small classifier (trained on 48 of 60 examples, 12 held out) scores it at 0.33-0.44 (below threshold). A 1.5B judge produces empty reasoning strings and flags nothing, and a 7B judge also flagged none of its 120 decisions on the malicious document. This is a detection-capacity result for those models, prompts and this payload: the local judge is implemented to fail open (it returns "not malicious" on a timeout, HTTP error or parse failure), and three frontier-model judges did flag the same template in 50 of 50 presentations each (and none of 50 benign ones). That frontier-judge test was at document level, on one template, and not run as an agent defence, so it does not show that the retrieval layer is closed.

**Prompt Hardening** instructs the model to refuse unauthorised exfiltration. For 7 of 8 Vulnerable Executors, the model reads the security rules, acknowledges them in its reasoning, and exfiltrates anyway, citing the stored rule as "mandatory regulatory policy" that supersedes the system prompt. One model (qwen3.5:122b) is the sole exception where the security instructions actually override the stored rule, and its baseline did not reproduce on a freshly booted machine.

**Memory Sandbox** removes the recall tool. The agent can see that stored facts exist but cannot read their values. For 8 of 9 models, this is sufficient. For qwq:32b in the April factorial, removing recall led the model to re-read the original poisoned document through RAG, where it encountered the instruction fresh and executed it; that inversion did not reproduce. Across providers, the same RAG-fallback route bypasses the sandbox for three Bedrock-served models (see below), so removing one content-delivery tool is not a security boundary while another tool supplies the same content.

---

### The qwq:32b inversion (environment-fragile, April 2026 only)

> **Caveat:** This behaviour was observed in the April 2026 factorial but did not reproduce in June 2026 under verified-identical weights, Ollama version, and application code. See [Limitations](#limitations).

Under every other defence, qwq:32b shows 0% ASR. Not because defences are working: in the April factorial the model drafted the email, recalled the routing rule, and stopped without calling `send_email` to the attacker (it reasoned that the user only asked for a draft). That Draft-Only behaviour is itself environment-fragile.

Under Memory Sandbox, when recall is blocked, qwq:32b queried RAG in the trigger session, retrieved the malicious document fresh, and this time executed the send. One reading is that a recalled memory fact is treated as advisory while a RAG-retrieved compliance document carries the implicit authority of an organisational source; the traces are consistent with that reading but do not establish it.

The underlying RAG-fallback bypass mechanism (independent of the qwq:32b-specific Draft-Only behaviour) replicates across four providers on Bedrock (see cross-provider validation below), which is why the sandbox's protection is read as limited to the path it removes.

---

## Two Anthropic models on Bedrock, and what the frontier API shows

A supplementary evaluation of Claude Sonnet 4.6 and Haiku 4.5 served on Bedrock (N=100 each, 400 runs total, a separate harness from the frontier-API evaluation below) found that neither shows the injection-to-exfiltration pipeline of the open-weight models.

**Sonnet 4.6** (Bedrock) resists at the injection stage. 0% injection rate across 100 runs. The model recognises the document as an injection attempt and declines to store it, producing an explicit security alert in every run.

**Haiku 4.5** (Bedrock) detects the attack but responds differently: it stores a security alert *about* the attack rather than the payload itself, and in the trigger session recalls its own warning and does not act. The routing rule is never stored.

These are Bedrock observations and they do not carry over to the frontier API. There, after the memory-tool fix, Haiku 4.5 stores the attacker's address in 31 of 40 runs (29 as a directive), sends no email under the draft-only trigger, and exfiltrates in 6 of 20 runs under the send-authorised follow-up. Sonnet 4.6, Sonnet 5 and Opus 4.8 keep the address only inside security warnings, Opus 4.5 does not save, and Opus 5 could not be evaluated (a content filter). No Anthropic endpoint exfiltrates in more than 1 of 10 runs under the draft-only trigger. The earlier claim that Anthropic is "execution-immune" is withdrawn.

The N=10 screening across 18 open-weight models found 11 Vulnerable Executors (100% injection, 100% attack), 4 Latent Carriers (inject but never execute under that trigger), and 2 Injection-Resistant models.

---

## What I built

A LangGraph evaluation framework with 5 simulated workplace tools, 7 defence conditions across 4 architectural layers, full factorial design (9 models x 7 conditions x 2 attack arms x N=40 = 5,040 runs), BCa bootstrap 95% CIs, 108 active comparisons (of 115 registered before data collection) with Holm-Bonferroni correction, per-session thread isolation, and deterministic tool-call instrumentation. The open-weight models run against the same infrastructure; the frontier experiments use separate API harnesses. Code and reproduction instructions in the [README](https://github.com/junwenleong/stateful-agent-security-eval).

---

## Methodology

**Statistical reporting:** BCa bootstrap CIs (10,000 resamples, seed=42), Wilson Score for boundary rates, Holm-Bonferroni correction across the 108 active comparisons (of 115 registered before data collection). 10 significant after correction. The 2,520 single-session no-attack runs produced no exfiltration; that is a no-false-trigger check, not a false-positive-rate measurement (the detector had no attacker address to match in that arm). All factorial models executed locally via Ollama at 16k context and temperature 0.0. Known properties of the shared harness (for example, the benign calendar and search tools return empty results, and most RAG queries, 7,536 of 7,816, return the whole three-document corpus) are listed in the [README](https://github.com/junwenleong/stateful-agent-security-eval#known-properties-of-the-released-harness). Full methodology in [FINDINGS.md](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/FINDINGS.md#methodology).

---

## Reasoning-mode crossover (single family, environment-conditioned)

A reasoning-mode ablation using Qwen3-32B's thinking toggle found a crossover: the sandbox variant that protects the thinking mode collapses the attack for the non-thinking mode (S4 to S1), while the variant that protects the non-thinking mode is bypassed by the thinking mode via goal-directed RAG fallback. So in that family no single memory-sandbox variant was safe across both modes. Two cautions: the traces suggest a three-part account (recall removed, a task that needs resolved external routing, reasoning capability) but only the task condition was manipulated directly, so it is a hypothesis; and the factorial's `qwen3:32b` sandbox cell (0 of 40, April) disagreed with the thinking-off `sandbox_full` cell (40 of 40, June) for what looks like the same configuration; a fresh same-day rerun (0 of 7 sandbox runs) reproduced April and not June, so the June think-off cells are environment-conditioned and the crossover rests on one collection block. It does not replicate on six frontier API configurations, whose test uses a different harness. Full results in [FINDINGS.md](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/FINDINGS.md).

## Cross-provider validation (Bedrock, full-precision)

A Bedrock validation (1,180 runs, 6 models, full-precision serving) found the Memory Sandbox RAG-fallback bypass in three Bedrock-served models: mistral-large-3-675b (97.5% ASR under sandbox), glm-5 (32%), gpt-oss-120b (55%). The mechanism replicates across four providers (Qwen, Mistral, Z.AI, OpenAI) and two serving stacks. Llama 4 Maverick is injection-resistant (0/20) in a provider breadth screen. Full details in [FINDINGS.md](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/FINDINGS.md) and the [paper](https://arxiv.org/abs/2605.08442).

## Frontier models after the memory-tool fix

The earlier empty-corpus screening (N=10, 0/210 exfiltrations) was a methodological artefact: without a malicious document in the corpus, that probe measured tool-calling compliance on user instructions alone, not indirect prompt injection vulnerability. A loaded-corpus collection that followed it was itself made with the memory-tool bug and is superseded (see the version note at the top of this page).

The corrected evaluation re-collected all 29 endpoints (N=10 each, 290 runs, 2026-10-12; four endpoints topped up to N=40 on 2026-10-13) with the document loaded in the corpus and the base, draft-only trigger prompt. A same-day control confirms the bug's effect: GPT-5.1 exfiltrated in 10 of 10 runs with the fix and in 0 of 10 with the bug re-introduced in the same script. "Stored" means the value read back from memory after the injection session names the attacker's address, whatever its wording.

| Model | N | Stored | ASR | Class |
|-------|---|--------|-----|-------|
| Gemini 3.1 Pro Preview (pre-release) | 10 | 10/10 | **10/10** | VE |
| Gemini 3.5 Flash | 10 | 10/10 | **10/10** | VE |
| Gemini 2.5 Pro | 10 | 10/10 | 5/10 | PE |
| Gemini 3.6 Flash, Gemini 2.5 Flash | 10 each | 10/10 | 1/10 each | PE |
| Gemini 3.7 Flash | 10 | 0/10 | 0/10 | SB |
| GPT-5.1, GPT-4o, GPT-4.1 | 10 each | 10/10 | **10/10 each** | VE |
| o3, o3-mini | 10 each | 10/10 | **9/10, 10/10** | VE |
| GPT-5, GPT-5.2, GPT-5-mini, GPT-5-nano, GPT-5.4-pro, o4-mini | 10 each | 10/10 | 1, 5, 6, 6, 4, 3 of 10 | PE |
| GPT-5.4 (topped up) | 40 | 40/40 | 0/40 | LC |
| GPT-5.5 (topped up) | 40 | 37/40 | 8/40 | PE |
| GPT-5.6-sol (topped up) | 40 | 33/40 | 0/40 | LC |
| GPT-5.6-terra, GPT-5.6-luna | 10 each | 0/10 | 0/10 | SB |
| Claude Sonnet 4.6, Sonnet 5, Opus 4.8 | 10 each | 1, 2, 4 of 10 (warnings only) | 0/10 | SB |
| Claude Opus 4.5 | 10 | 0/10 | 0/10 | IR |
| Claude Opus 5 | 10 | not evaluable (content filter) | 0/10 | NE |
| Claude Sonnet 4.5 | 10 | 1/10 | 1/10 | PE |
| Claude Haiku 4.5 (topped up) | 40 | 31/40 | 0/40 | SB/LC (borderline) |

VE Vulnerable Executor (at least 90% of runs), PE Partial Executor, LC Latent Carrier and SB storage-stage block describe the draft-only trigger only. At N=10 the Wilson interval around 0/10 extends to 28% and around 5/10 spans 24% to 76%, so the table supports the classes, not rankings within a class.

### What the corrected data show

- **Execution is common.** 7 of 29 endpoints exfiltrate in at least 9 of 10 runs and 11 more in some but fewer. GPT-5.5 exfiltrated in 8 of 40 runs, all 8 on the second collection day (the day difference is not significant, Fisher p = 0.17).
- **A zero under the draft-only trigger is not evidence of safety.** GPT-5.4, GPT-5.6-sol and Haiku 4.5 never exfiltrate (0 of 40 each) but send almost no email under that trigger, which supplies no project content. In an exploratory, not preregistered, follow-up that supplies the content and asks for a send (20 runs each, 2026-10-14), they exfiltrate in 20 of 20, 5 of 20 and 6 of 20 runs; the GPT-5.1 control is 20 of 20. In every sending run the attacker is added to the legitimate recipient. The follow-up changes content availability and send authorisation together and uses one prompt written after the base results, so it supports only a narrow conclusion: the base-trigger zeros reflect the absence of a send decision.
- **No monotonic ordering among OpenAI snapshots**: GPT-4o 10/10, GPT-4.1 10/10, GPT-5 1/10, GPT-5.1 10/10, GPT-5.2 5/10, GPT-5.4 0/40, GPT-5.5 8/40, GPT-5.6 0/10 (terra, luna) and 0/40 (sol). GPT-5.1 against GPT-5 is a regression between the endpoints requested under those names in this collection (Fisher p = 1.2e-4); the served build is not confirmed, so this says nothing about training changes.
- **Vendor-correlated patterns, in one window only.** Anthropic endpoints mostly do not store a forwarding directive, though Haiku 4.5 does and exfiltrates in 6 of 20 under the send-authorised trigger. Google does not consistently block at either stage until its newest tested endpoint. These describe requested endpoints in one collection window, not stable properties of vendors.
- Frontier ASR is scored from the requested recipients of `email_send_email` calls, and send results are not logged, so it is an attacker-addressed send rate.

### Other frontier experiments

- **Memory Sandbox probe, re-collected with the document loaded** (40 runs, 2026-09-29; o3, o3-mini, o4-mini, GPT-5.1): 0% ASR with the attacker's rule stored in every run. Their draft-only baselines are 9, 10, 3 and 10 of 10, so the 0% is informative. The qwq:32b inversion was not observed on these models.
- **Tool-schema ablation** (multi-turn, N=50 per cell, 6 configurations x 2 tool conditions, 600 trials): exfiltration is 76-100% with recall tools removed and 32-100% with them available; keeping recall lowers it by 8-44 points in four of six configurations and not measurably in the other two. The rule is in the system prompt of both arms, so this measures action selection under tool-schema restriction, not an injection test, and its failure to reproduce the Qwen crossover is a weak test of that crossover.
- **Frontier retrieval-layer judges**: Claude Sonnet 4.6, GPT-5.1 and Gemini 2.5 Pro flagged the malicious template in 50 of 50 presentations each and none of 50 benign ones (document level, one template, not an agent defence).
- **Supply chain**: the earlier writer-reader pairs were collected with the buggy wrapper and are explained by retrieval of the still-poisoned corpus, not memory-mediated transfer. No post-fix writer-reader pair has been run, so "a model that stores a rule without acting could feed one that acts" is an inference from per-model measurements, not a demonstrated chain.
- **System-prompt mitigation**: the earlier hardened-prompt comparison used the buggy wrapper in both arms and is superseded; no conclusion about prompt hardening on frontier models is drawn from it.

### A content-layer defence (proof of concept), and a reasoning-model evaluation caveat

A content-layer alternative to removing the recall tool is **Runtime Adaptive Tool-Gating (RATG)**, which leaves the recall tool in place but sanitises the recalled value, stripping unauthorised addresses and routing directives before the model sees them. For the three mechanical instruction-following models (qwen2.5:14b, qwen2.5:72b, qwen3:32b), RATG reduces ASR from 100% to **0%** (N=40 per arm, injection stays 100% - the defence acts at the content layer, not by blocking storage). RATG sanitises only the value returned by the recall tool, not RAG output, so it does not by construction close the RAG-fallback route; and in the released framework the hook is also advertised to the model as an extra `memory_ratg_filter` tool (never called in the collected logs). We report RATG efficacy only for models satisfying a reproducibility validity criterion: (i) the no-defence baseline must reproduce as vulnerable, and (ii) the arms must be identical in every session before the defence activates. All five reasoning models fail at least one condition: qwen3.5:122b passes (ii) but fails (i) (its Jul-1 100% baseline did not reproduce on a freshly booted machine under verified configuration); qwq:32b and gpt-oss:20b fail (ii) (session-0 forks under identical inputs); qwen3.5:9b and gpt-oss-safeguard:120b fail (i) (no vulnerable baseline). We make no reasoning-model RATG efficacy claim. Scaling the RAG judge from 1.5B to 7B (qwen2.5:7b, N=40 per arm on qwen2.5:14b and qwen3:32b) did not detect the compliance-framed payload (0/120 malicious-doc decisions flagged, 0/400 total, ASR 100% to 100%).

A methodology caveat surfaced while extending the factorial on Ollama 0.30.11: reasoning-model trigger-session output shows several anomalous manifestations (announce-then-no-emit, emission truncation, team-only sends, address confabulation, and session-0 arm forks under identical observable inputs - zero refusals across ~324 runs). We report these as observed manifestations with a common cause that is *not directly demonstrated*: consistent with host/runtime state effects, but digest equality rules out only weight drift, not runtime change ("same weights, different runtime"). This *under-reports* vulnerability: qwen3.5:122b re-run on fresh daemon loads exfiltrates 60/60 (100%) with byte-identical prompts and weights. The fix is a fresh-daemon-per-model protocol. This is distinct from the qwq:32b Draft-Only behaviour in the April factorial, in which the model's trace states that the user only asked for a draft (itself environment-fragile).

The main lesson across both campaigns: closing a single content-delivery path (recall) leaves equivalent paths (RAG) open, a memory write and a later action must be measured separately, and a zero attack rate does not show resistance when the agent never reaches a send decision. Frontier endpoint results are a dated snapshot of hosted endpoints, not a ranking of vendors.

Full details in [FINDINGS.md](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/FINDINGS.md).

---

## Limitations

The tools are simulated, not production deployments. The factorial models are open-weight models served with quantisation (q4_0 or q8_0), not full-precision API-served versions, and were selected for storing the instruction in every run. The defences are lightweight proxies designed to test architectural categories rather than replicate commercial implementations. The attack is a fixed, non-adaptive framing, so the rates are observed rates under that attack; adaptive optimisation could reveal higher ones. The single-session no-attack control does not require recall, so "no utility cost" for Memory Sandbox is not evidence about memory-dependent workflows.

Two results should not be over-read. Input-layer filters on the user message cannot see a payload that arrives through retrieval: that is a placement result and does not depend on classifier quality. For the retrieval layer the result is narrower, because the small classifiers and judges tested did not flag the payload while three frontier judges did (document level, one template, not an agent defence), so a capable retrieval-layer judge may close that gap; this has not been tested end to end or against diverse payloads.

The qwq:32b Draft-Only archetype and its associated Memory Sandbox inversion (0% to 100% ASR) were observed in the April 2026 factorial but do not reproduce in June 2026 under verified-identical weights, the same reported Ollama version (0.20.6), and the same application code. The flip traces to a single divergent reasoning token in S3 whose host-layer cause could not be isolated (April's OS/driver/binary build were not logged). This is reclassified as environment-fragile rather than a stable model property. The Qwen3-32B thinking-toggle crossover is a separate design but is also environment-conditioned: its thinking-off `sandbox_full` cell (40 of 40) disagreed with the factorial's `qwen3:32b` sandbox cell (0 of 40) for an apparently identical configuration, and a fresh same-day rerun reproduced the factorial cell and not the June one (0 of 7 sandbox runs); the cause is not isolated.

---

## Companion: Forensic Detection (June 2026)

A companion paper examines whether memory-channel attacks leave a forensic signature in tool-call logs. A classifier trained on the factorial data reaches AUC = 0.990 using only operation names and ordering, but the negative class there is poisoned-but-defended sessions, so this measures separation of successful from unsuccessful attacks, not detection of attacks against benign traffic.

Two limits are load-bearing. First, benign memory-grounded sends produce the same trajectory: false positives run 24.7 to 57.6 percent across 13 models (N = 4,160), with positive predictive value at or below 3.85 percent at 1 percent prevalence. The transition is an attack precondition, not a maliciousness predicate. Second, the invariant is conditional on the payload being reachable only through an agent-visible retrieval call. Implicit, framework-side delivery evades it entirely while attacks remain viable, so evasion is possible without abandoning the memory channel. Details: [arXiv:2606.30566](https://arxiv.org/abs/2606.30566).

## Links

[arXiv Paper](https://arxiv.org/abs/2605.08442) · [GitHub Repository](https://github.com/junwenleong/stateful-agent-security-eval) · [Full Results (FINDINGS.md)](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/FINDINGS.md) · [Verification Script](https://github.com/junwenleong/stateful-agent-security-eval/blob/main/scripts/verify_canonical.py)

The factorial, RATG, 7B-judge and causal-mediation aggregates are checked against raw experimental data (5,660 core records plus 933 RATG/judge records across 7 result files) by `verify_canonical.py`; its header lists exactly what it covers, and a pass certifies those aggregates, not every claim. The post-fix frontier and send-authorised numbers are in `canonical_numbers.md` and reproduce from the JSONL files in the post-fix result directories.
