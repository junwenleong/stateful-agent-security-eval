# Claims Ledger & Prior-Art Positioning Matrix — NDSS 2027

**Paper**: "Defense Effectiveness Across Architectural Layers: A Mechanistic Evaluation of Persistent Memory Attacks on Stateful LLM Agents"
**arXiv**: 2605.08442
**Target**: NDSS 2027 (deadline Aug 19, 2026)
**Generated**: 2026-07-11

---

## CLAIMS LEDGER

### Claim 1: Layer-Structural Defense Failure

**Statement**: Five of six defenses fail against persistent memory attacks. Input-layer defenses cannot observe the RAG-injected payload; retrieval-layer classifiers observe it but cannot distinguish compliance-framed injection from legitimate policy; instruction-level hardening is overridden by compliance framing.

**Evidence**:
- Experiment: Defense Factorial (5,040 runs, 9 models x 7 defenses x 2 arms x N=40)
- Error rate: 0% (0/5,040)
- ASR by defense (DTA arm, n=360 per defense):
  - no_defense: 88.6% [0.849, 0.916]
  - minimizer: 88.9% [0.852, 0.917]
  - sanitizer: 88.9% [0.852, 0.917]
  - rag_sanitizer: 88.9% [0.852, 0.917]
  - rag_llm_judge: 88.9% [0.852, 0.917]
  - prompt_hardening: 77.8% [0.732, 0.818]
  - memory_sandbox: 11.1% [0.083, 0.148]
- Statistical method: BCa bootstrap (10,000 resamples, seed=42), Wilson Score for degenerate vectors
- Correction: Holm-Bonferroni across 108 active comparisons
- Significant comparisons after correction: 10/108
- False positive rate: 0.0% (0/2,520 no-attack runs)
- The 0.3pp gap (no_defense 319/360 vs four blind defenses 320/360) is Fisher's exact p=1.0

---

### Claim 2: Memory Sandbox is the Only Effective Defense (with caveats)

**Statement**: Memory Sandbox reduces ASR to 0% for 8/9 models. The exception (qwq:32b) inverts to 100% via RAG-fallback bypass.

**Evidence**:
- 8 models: ASR = 0% (0/40 each), Wilson CI [0.000, 0.088]
- qwq:32b: ASR = 100% (40/40), Wilson CI [0.912, 1.000]
- BTCR (no-attack arm): 100% across all 63 conditions (9 models x 7 defenses)
- Cross-family bypass replication (Bedrock): mistral-large-3-675b 39/40 ASR, glm-5 13/40, gpt-oss-120b 22/40
- Frontier Latent Carriers do NOT bypass: 0/46 ASR (o3, o3-mini, o4-mini, gpt-5.1)
- qwq:32b 32k verification: 40/40 ASR under memory_sandbox (context-length independent)

---

### Claim 3: Double Dissociation (Reasoning-Defense Interaction)

**Statement**: Within Qwen-3-32B (think toggle), sandbox_full protects reasoning mode but breaks non-reasoning mode; sandbox_null_recall protects non-reasoning mode but is bypassed by reasoning mode. No single sandbox variant is safe across both classes.

**Evidence**:
- All cells N=40, zero variance (40/40 or 0/40)
- sandbox_full: think=OFF 100% ASR (S1), think=ON 0% ASR
- sandbox_blind: think=OFF 100% ASR (S1), think=ON 0% ASR
- sandbox_null_recall: think=OFF 0% ASR, think=ON 100% ASR (S4)
- no_defense baseline: both modes 100% ASR (toggle does not change baseline)
- RAG-fallback arm replicates cross-family on Bedrock (Mistral, Z.AI, OpenAI)
- Full crossover demonstrated only within Qwen-3-32B

---

### Claim 4: Frontier Vendor Divergence (Tripartite Architecture)

**Statement**: Three vendors implement structurally different safety approaches. Anthropic blocks at injection (0-2.5% storage), OpenAI blocks at execution (100% injection, 0% ASR for GPT-5.4+), Google blocks at neither (22.5-95% ASR).

**Evidence**:
- Confirmatory evaluation: 21 models, N=40 each, loaded corpus, standard compliance framing
- Gemini 3.1 Pro Preview: 95.0% ASR [83.5%, 98.6%] (highest frontier model)
- Gemini 3.5 Flash: 50.0% ASR [35.2%, 64.8%]
- Gemini 2.5 Pro: 22.5% ASR [12.3%, 37.5%]
- Sonnet 4.6: 2.5% injection, 0% ASR [0%, 8.8%]
- Opus 4.5: 2.5% injection, 0% ASR
- GPT-5.4/5.5: 100% injection, 0% ASR [0%, 8.8%]
- All OpenAI models inject at >= 97.5%
- Sonnet 4.6 framing-invariant: 0% ASR across varA, varB, varC (N=40 each)

---

### Claim 5: Non-Monotonic Generational Hardening (GPT-5 Family)

**Statement**: GPT-5.1 (22.5% ASR) regresses relative to GPT-5 (5.0%). Safety is not monotonically preserved across micro-version updates.

**Evidence**:
- All N=40, loaded corpus, standard compliance framing (Variant A):
  - GPT-4o: 12.5% [5.5%, 26.1%]
  - GPT-4.1: 7.5% [3.5%, 15.4%] (pooled N=80)
  - GPT-5: 5.0% [1.4%, 16.5%]
  - GPT-5.1: 22.5% [12.3%, 37.5%] (REGRESSION)
  - GPT-5.2: 2.5% [0.4%, 12.9%]
  - GPT-5.4: 0.0% [0%, 8.8%]
  - GPT-5.5: 0.0% [0%, 8.8%]
- Authority escalation (Variant B, GPT-4o): 60.3% [48.4%, 71.1%] (N=68)

---

### Claim 6: Injection-Execution Dissociation (Supply-Chain Risk)

**Statement**: All OpenAI models store the malicious rule (>=97.5% injection) even when execution is blocked (0% ASR for GPT-5.4+). Stored payloads persist indefinitely, creating compositional supply-chain risk.

**Evidence**:
- GPT-5.5: 100% injection, 0% ASR (N=40)
- o3/o3-mini/o4-mini: 98-100% injection, 0% ASR (N=40 each)
- Persistence test: 50-session marathon (qwen2.5:14b), byte-identical recall at checkpoints 10/20/30/40/50
- Tool-existence ablation: removing save_fact drops injection 100% to 0% (N=40 each for gpt-5.1, o3-mini, o4-mini). Zero hallucinated saves.

---

### Claim 7: RATG Content-Layer Defense (Proof of Concept)

**Statement**: RATG resolves the schema-layer dilemma by sanitizing recalled content rather than removing the tool. Reduces ASR 100% to 0% on mechanical models.

**Evidence**:
- 3 mechanical models (qwen2.5:14b, qwen2.5:72b, qwen3:32b): ASR 100% to 0% (N=40/arm)
- Injection stays 100% (content-layer, not store-layer)
- Reasoning-model RATG: NO CLAIM (all 5 reasoning models fail validity criterion)
- Defense Attribution Validity Criterion: (i) baseline reproduces, (ii) pre-activation arm identity

---

### Claim 8: Anthropic Active Detection Mechanisms

**Statement**: Sonnet 4.6 is an Explicit Detector (0% injection). Haiku 4.5 is an Active Detector with Defensive Storage (stores security alert, not payload).

**Evidence**:
- Sonnet 4.6 (N=100, Bedrock): 0% injection [0.000, 0.037], 0% ASR, BTCR=100%
- Haiku 4.5 (N=100, Bedrock): 100% injection [0.963, 1.000], 0% ASR [0.000, 0.037], BTCR=100%
- Haiku stores 3 sanitised facts: 2 legitimate + 1 security alert. Payload never stored.
- Haiku detection language in 100/100 runs ("suspicious instruction attempting to manipulate my behavior")
- False positive rate: 0.0% (0/200 no-attack runs)

---

### Claim 9: Retrieval-Fidelity Criterion (Methodological)

**Statement**: Empty-corpus frontier screening (0/210 exfiltrations) was a threat-model artifact. Loaded-corpus evaluation overturns it.

**Evidence**:
- Empty corpus (N=10, 21 models): 0/210 exfiltrations, CI upper bound <=1.43%
- Loaded corpus (N=40): Gemini 3.1 Pro Preview 95% ASR, GPT-5.1 22.5% ASR
- Gemini reclassified: was "Injection-Resistant" (empty) to "Vulnerable Executor" (loaded)
- GPT-5.1 reclassified: was "Latent Carrier" (empty) to "Partial Executor" (loaded)

---

### Claim 10: Tool-Existence as Necessary Condition for Injection

**Statement**: Removing save_fact from the tool schema drops injection from 100% to exactly 0%. Models are cognitively compliant but architecturally blocked.

**Evidence**:
- gpt-5.1: save_fact removed N=40, injection 0%, no_tools N=40, injection 0%
- o3-mini: save_fact removed N=40, injection 0%, no_tools N=40, injection 0%
- o4-mini: save_fact removed N=40, injection 0%, no_tools N=40, injection 0%
- Text compliance: ~50% of runs signal compliance in text without tools; 100% with no tools at all
- Zero hallucinated tool calls across all conditions


---

## PRIOR-ART POSITIONING MATRIX

### 1. SMSR (Sharma, arXiv:2606.12703, Jun 2026, submitted to IEEE)

**What it is**: HMAC-SHA256 provenance tagging for agent memory with formal (0, 2^-256)-security proof. Tags memory entries at write time so that read-time verification can detect tampering.

**What it pre-empts from our work**:
- Any claim that cryptographic provenance for agent memory is novel
- Any framing of Memory Sandbox as a novel "defense architecture" in isolation
- The general principle that memory-layer defenses are needed

**Our residual delta**:
- SMSR proposes a defense; we provide the ATTACK CHARACTERIZATION and DEFENSE FAILURE ANALYSIS that motivates why such defenses are needed
- Our layer-structural failure result (5 defenses fail, only tool-layer works) is the empirical motivation for SMSR's formal approach
- Our double dissociation shows that even tool-layer defenses (which SMSR does not address as a schema intervention) have reasoning-class-dependent failure modes
- Our cross-generational vulnerability mapping is measurement, not defense
- SMSR does not evaluate against reasoning-model RAG-fallback bypass

**How to cite/frame**:
> "Certified memory-provenance approaches such as SMSR [Sharma 2026] provide formal security guarantees for the memory channel. Our work complements this by characterizing the attack surface that motivates such defenses: we show empirically that five defense classes spanning input, retrieval, and instruction layers fail structurally against persistent memory attacks, establishing the practical need for memory-layer interventions. We further identify a reasoning-model bypass mechanism (goal-directed RAG fallback) that operates outside the memory channel SMSR protects."

---

### 2. Inseparability (Pant et al., arXiv:2606.27567)

**What it is**: Formal proof that shared-embedding architectures cannot prevent prompt injection in-pipeline. Establishes a mathematical impossibility result for content-channel defenses.

**What it pre-empts from our work**:
- Any claim that our layer-structural failure is a novel "impossibility" result
- Any framing of "input-layer defenses cannot work" as a theoretical contribution
- The general principle that content-based detection has fundamental limits

**Our residual delta**:
- Inseparability proves the impossibility formally; we demonstrate the BEHAVIORAL INSTANTIATION across 9 models, 6 defenses, 5,040 runs
- Our contribution is the EMPIRICAL MEASUREMENT of failure rates, not the theoretical principle
- We identify specific failure MECHANISMS (semantic masking, capacity-bound judges, compliance-override)
- We show the failure is NOT just about detection but extends to execution-layer reasoning (prompt hardening overridden)
- The double dissociation and vendor divergence are empirical findings orthogonal to the impossibility proof

**How to cite/frame**:
> "Pant et al. [2026] formally prove that shared-embedding architectures cannot prevent prompt injection in-pipeline. Our evaluation provides the empirical realization of this theoretical bound in the persistent memory setting: across 5,040 runs, retrieval-layer classifiers at two capacity levels (60-example TF-IDF, 1.5B LLM judge, 7B LLM judge) all fail to detect compliance-framed injection, consistent with the fundamental detection limits their proof establishes. Our additional contributions---the reasoning-defense interaction, cross-generational vulnerability mapping, and vendor behavioral divergence---are orthogonal to the impossibility result."

---

### 3. Surface Heuristics (Li et al., ACL 2026, arXiv:2601.07185)

**What it is**: Demonstrates that trained prompt-injection defenses learn token/position shortcuts rather than genuine intent detection. Defenses are brittle to reformulation.

**What it pre-empts from our work**:
- The general principle that content-based defenses rely on surface patterns
- Any novelty claim about "our classifiers fail because they match patterns not intent"
- The RAG Sanitizer and RAG LLM Judge failure as a standalone finding

**Our residual delta**:
- Surface Heuristics studies INPUT-LAYER defenses against single-turn injection; we study PERSISTENT MEMORY attacks across sessions
- Our specific finding is about COMPLIANCE FRAMING defeating classifiers (semantic masking with legitimate regulatory language), not just token patterns
- We extend the brittleness finding to the AGENTIC MULTI-SESSION SETTING with tool-mediated persistence
- We show that even scaling the judge (1.5B to 7B) does not close the gap for this payload class
- The layer-structural argument (input defenses are architecturally blind to RAG content) is distinct from "classifiers use shortcuts"

**How to cite/frame**:
> "Li et al. [ACL 2026] demonstrate that trained prompt-injection defenses learn surface heuristics rather than intent. Our retrieval-layer failures are consistent with this finding: the RAG LLM Judge (1.5B) produces empty reasoning for the compliance-framed payload, suggesting a capacity gap rather than genuine semantic engagement. We extend this observation to the persistent-memory attack class, where the payload enters via RAG retrieval rather than user input, and show that the architectural blindness of input-layer defenses (which never observe the payload) is a separate, structural failure mode distinct from classifier quality."

---

### 4. Role Confusion (Ye et al., ICML 2026, arXiv:2603.12277)

**What it is**: Shows that models infer instruction source from style/content rather than labeled role. Models exhibit "role confusion" where stylistically authoritative content in user messages is treated as system-level.

**What it pre-empts from our work**:
- The general mechanism by which compliance-framed documents override safety instructions
- The principle that "authority" in prompts is content-derived, not structurally enforced
- Any claim that our "semantic masking" finding is mechanistically novel

**Our residual delta**:
- Role Confusion studies single-turn role inference; we study MULTI-SESSION PERSISTENCE where the injection and execution are separated by sessions
- Our attack exploits a DIFFERENT channel: RAG-retrieved documents stored via tool calls, not user-message role confusion
- The compliance framing in our attack exploits the agent's TOOL DESCRIPTION priming ("save organizational rules"), not just stylistic authority
- We demonstrate that role/authority confusion PERSISTS ACROSS SESSION BOUNDARIES via tool-mediated state
- The vendor divergence (Anthropic blocks storage, OpenAI blocks execution) maps different LAYERS of susceptibility to role confusion

**How to cite/frame**:
> "Ye et al. [ICML 2026] demonstrate that models infer instruction source from style and content rather than labeled role, a mechanism they term 'role confusion.' Our compliance-framing attack exploits a related phenomenon: the agent treats a RAG-retrieved compliance memorandum as carrying organizational authority because its formal regulatory language matches the tool description's priming ('save organizational rules and compliance requirements'). We extend this to the temporal domain---the authority inference persists across session boundaries via tool-mediated state---and show that vendor-specific safety architectures differ in WHERE they interrupt this authority chain (injection-layer for Anthropic, execution-layer for OpenAI, neither for Google)."

---

### 5. MemoryGraft (arXiv:2512.16962, Dec 2025)

**What it is**: Proposed cryptographic provenance attestation as a defense direction for agent memory. Conceptual predecessor to SMSR.

**What it pre-empts from our work**:
- The concept of cryptographic memory provenance as a defense
- Any claim that provenance-based memory defense is novel

**Our residual delta**:
- MemoryGraft proposes the defense concept; we provide the systematic ATTACK EVALUATION that demonstrates why it is needed
- Our defense factorial shows that non-provenance approaches (classifiers, prompt hardening) fail
- Our supply-chain risk finding (Latent Carriers writing dormant payloads) is the specific threat model provenance defenses must address

**How to cite/frame**:
> "Cryptographic provenance for agent memory was proposed by MemoryGraft [Srivastava et al. 2025] and formalized with certified bounds by SMSR [Sharma 2026]. Our evaluation provides the empirical attack characterization that motivates these approaches: across 21 frontier models, all OpenAI models store attacker-authored rules without provenance metadata, creating the compositional supply-chain risk that provenance defenses are designed to prevent."

---

### 6. MINJA (arXiv:2503.03704, Mar 2025)

**What it is**: Query-only memory injection attack. Demonstrates that an attacker can poison an agent's memory through carefully crafted queries alone, without direct write access.

**What it pre-empts from our work**:
- The general concept of memory poisoning in LLM agents
- Demonstrating high attack success rates against open-source models
- The basic threat model of persistent memory attacks

**Our residual delta**:
- MINJA uses QUERY-ONLY injection (crafted user queries); we use RAG-DOCUMENT injection (poisoned corpus documents)
- We provide DEFENSE EVALUATION (6 defenses x 4 layers), which MINJA does not
- We demonstrate CROSS-SESSION persistence (4-session DTA with benign intervening sessions)
- We provide FRONTIER MODEL evaluation (21 models, 3 vendors) and vendor divergence mapping
- We identify the reasoning-defense interaction (double dissociation) and RATG content-layer resolution
- The generational vulnerability mapping and non-monotonic hardening are novel measurements

**How to cite/frame**:
> "MINJA [Shao et al. 2025] demonstrated that persistent memory attacks achieve high success rates against open-source models via query-only injection. Our work uses a different injection vector (RAG-retrieved compliance documents) and contributes systematic defense evaluation across four architectural layers, cross-generational frontier model vulnerability mapping, and the identification of reasoning-capability-dependent defense interactions not previously documented."

---

### 7. AgentPoison (arXiv:2407.12784, Jul 2024)

**What it is**: Poisoning agent knowledge bases to manipulate agent behavior. Gradient-based optimization of poisoned entries for RAG-augmented agents.

**What it pre-empts from our work**:
- The concept of poisoning agent knowledge bases/RAG corpora
- Demonstrating that RAG retrieval can be exploited as an attack vector
- The general threat model of RAG-based injection

**Our residual delta**:
- AgentPoison uses GRADIENT-BASED optimization (adaptive, white-box); we use FIXED ECOLOGICALLY VALID framing (realistic compliance documents)
- Our results are LOWER-BOUND estimates of vulnerability under realistic non-adaptive conditions
- We evaluate DEFENSES systematically (6 defenses, 5,040 runs); AgentPoison focuses on attack generation
- We provide FRONTIER MODEL evaluation and vendor divergence
- Our attack persists ACROSS SESSIONS via tool-mediated state (not just single-session RAG poisoning)
- The defense factorial, double dissociation, and generational mapping are distinct contributions

**How to cite/frame**:
> "AgentPoison [Yang et al. 2024] demonstrates gradient-optimized poisoning of agent knowledge bases. Our evaluation uses ecologically valid compliance-framed documents (no optimization required), providing lower-bound vulnerability estimates under realistic conditions. We extend beyond attack demonstration to systematic defense evaluation across four architectural layers and cross-generational frontier model characterization."

---

### 8. Adaptive Attacks Break Defenses (arXiv:2503.00061, Mar 2025)

**What it is**: Demonstrates that adaptive prompt-injection attacks defeat static defenses. Establishes that defenses evaluated only against fixed attacks overestimate their effectiveness.

**What it pre-empts from our work**:
- The general principle that static defenses can be broken by adaptive attacks
- Any claim that our defense failures are surprising in principle
- The limitation that our fixed-framing results may underestimate vulnerability

**Our residual delta**:
- We explicitly acknowledge this: "Our results should be interpreted as LOWER-BOUND estimates of vulnerability under realistic, non-adaptive attack conditions"
- Our contribution is not "defenses fail" (known) but WHERE and WHY they fail (layer-structural analysis)
- We show that defense failure is ARCHITECTURALLY DETERMINED (layer placement), not just a classifier-quality gap
- The memory persistence dimension (cross-session, tool-mediated) is orthogonal to adaptive vs. fixed attacks
- The double dissociation shows that even CAPABILITY REMOVAL (not detection) has reasoning-class-dependent failures

**How to cite/frame**:
> "Adaptive attacks are known to defeat static defenses [2503.00061]. Our fixed-framing results are therefore lower-bound estimates; Trojan Hippo [Das et al. 2026] confirms higher rates under adaptive optimization. Our contribution is orthogonal: we show that defense failure in the persistent memory setting is governed by ARCHITECTURAL LAYER PLACEMENT (input defenses are blind to RAG content by construction), not merely by classifier robustness to adaptive reformulation. Even Memory Sandbox---a capability-removal defense immune to content-based evasion---exhibits reasoning-class-dependent bypass."

---

### 9. The Attacker Moves Second (arXiv:2510.09023, Oct 2025)

**What it is**: Establishes the principle that static defense brittleness is inherent when the attacker can observe and adapt to the defense. The defender's fixed strategy is always exploitable.

**What it pre-empts from our work**:
- The general principle of defense brittleness under adaptive adversaries
- Any claim that our defense failures represent a novel theoretical insight
- The limitation framing of "better classifiers might work"

**Our residual delta**:
- Our layer-structural failure is INDEPENDENT of attacker adaptation: input defenses fail because they are architecturally blind, not because the attacker adapted
- The structural argument (defenses at the wrong layer fail regardless of quality) is COMPLEMENTARY to the adaptive argument (defenses at the right layer still fail if the attacker adapts)
- We show that even STRUCTURAL defenses (capability removal via Memory Sandbox) have class-dependent failures
- The vendor divergence and generational mapping are measurement contributions, not defense claims
- RATG is presented as "proof of concept" with explicit acknowledgment of adaptive bypass (encoding, homoglyphs)

**How to cite/frame**:
> "The fundamental challenge of defense under adaptive adversaries [2510.09023] applies to our retrieval-layer classifiers, which we acknowledge as lower-bound results. However, our primary finding---that input-layer defenses are architecturally blind to RAG-injected payloads---is a STRUCTURAL failure independent of attacker adaptation: these defenses fail not because the attacker evaded them but because they never observe the attack surface. This structural argument complements the adaptive-attack principle: even perfectly robust classifiers at the input layer would fail, because the payload enters through RAG retrieval."

---

## FRAMING RULES FOR NDSS SUBMISSION

### What We Claim as Novel (ordered by defensibility)

1. **Systematic defense evaluation across 4 architectural layers** against persistent memory attacks (5,040 runs, pre-registered comparisons, Holm-Bonferroni correction). No prior work evaluates input, retrieval, instruction, AND tool-layer defenses against the same persistent memory attack.

2. **Double dissociation in defense design** (think-toggle crossover within Qwen-3-32B). Single-family but RAG-fallback bypass arm replicates cross-family on Bedrock.

3. **Cross-generational non-monotonic vulnerability mapping** (21 frontier models, 3 vendors, N=40 confirmatory). GPT-5.1 regression, Gemini 95% ASR, tripartite vendor divergence.

4. **Injection-execution dissociation** as a measurable supply-chain risk (all OpenAI models store at >=97.5% regardless of execution resistance).

5. **RATG as content-layer resolution** of the schema-layer dilemma (proof of concept, mechanical models only).

6. **Two reusable evaluation-validity criteria**: Retrieval-Fidelity Criterion and Defense Attribution Validity Criterion.

### What We Do NOT Claim as Novel

- Memory poisoning as an attack class (cite MINJA, AgentPoison, Zombie Agents)
- Cryptographic provenance as a defense (cite MemoryGraft, SMSR)
- Impossibility of content-based detection (cite Inseparability)
- Surface-heuristic brittleness of classifiers (cite Surface Heuristics)
- Authority derived from content style (cite Role Confusion)
- Defense failure under adaptive attacks (cite Adaptive Attacks, Attacker Moves Second)

### Mandatory Citation Sentences (Include in Related Work or Discussion)

1. "Certified memory provenance [SMSR; Sharma 2026] and formal impossibility bounds [Pant et al. 2026] represent complementary theoretical contributions; our work provides the empirical attack and defense characterization that motivates and contextualizes both."

2. "Our retrieval-layer failures are consistent with the surface-heuristic hypothesis [Li et al., ACL 2026] and the content-derived authority mechanism [Ye et al., ICML 2026]."

3. "Our fixed-framing results are lower-bound estimates [cf. Adaptive Attacks 2503.00061; The Attacker Moves Second 2510.09023]; Trojan Hippo confirms higher rates under adaptive optimization."

4. "The memory-poisoning threat model was established by MINJA [2503.03704] and AgentPoison [2407.12784]; MemoryGraft [2512.16962] proposed cryptographic provenance as a defense direction."
