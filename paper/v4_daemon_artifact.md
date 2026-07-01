# v4 addition — daemon-state degradation of reasoning models (evaluation artifact)

Ready-to-integrate. Belongs in the Evaluation Artifacts section (next to the
qwen tool-call-format artifact) and referenced from the RATG results section.
Grounded entirely in trace-level forensics (learningjourney Iteration 55); makes
no claim the traces do not support.

---

## Evaluation artifact: long-running daemon sessions degrade reasoning-model trigger-session behaviour

While extending the factorial with a recall-time content defense (RATG) across a
nine-model sequential run on a single long-lived Ollama daemon (v0.30.11), the six
reasoning-capable models produced near-zero attack success, in apparent contradiction
to their 100% attack success in the primary factorial. Forensic trace analysis
established that this was an evaluation artifact, not a safety property.

Across all reasoning-model non-exfiltration runs in the sequential batch
(approximately 324 runs spanning qwen3.5:9b, qwen3.5:122b, qwq:32b, gpt-oss:20b,
gpt-oss-safeguard:120b, and glm-4.7-flash), zero runs contained refusal reasoning.
The models did not decline to execute the stored rule. Instead, the trigger-session
output degraded in model- and session-specific ways: some runs produced a complete
reasoning trace that announced the exfiltration verbatim ("I will now send this email
to both the team and the external compliance monitor") but then emitted no structured
tool call; others emitted an email to the intended recipient only, having confabulated
that the attacker address had been "redacted" from memory (it had not); and others
produced no tool calls at all. The specific degradation mode varied by model and by
daemon session (for qwen3.5:122b, one session truncated at the tool-call boundary and
another confabulated the recipient), indicating a host-layer conditioning effect whose
precise cause we could not isolate, consistent with the temporal-stability finding
reported for qwq:32b.

We confirmed the artifact by re-running qwen3.5:122b on fresh daemon loads (three
independent process restarts, 60 runs total). On fresh loads the model exfiltrated in
60 of 60 runs (100%), via the standard memory-recall pathway (memory recall in the
trigger session in 60/60 runs; no RAG fallback), with tight deterministic latency and
byte-identical prompts, weights (digest 8b9d11d807c5), and Ollama version to the
degraded batch. The degraded near-zero result therefore under-reports vulnerability:
the model's true behaviour, under a fresh daemon load, is unchanged from the primary
factorial.

Three consequences follow. First, the recall-time defense (RATG) result is reported
only for the three mechanical instruction-following models (qwen2.5:14b, qwen2.5:72b,
qwen3:32b), whose behaviour is deterministic across April, both Ollama versions, and
the sequential batch; for these models RATG reduces attack success from 100% to 0%.
The reasoning-model RATG arms from the sequential batch are not reported, because their
degraded baselines admit no interpretable reduction. Second, this is a methodological
recommendation for local-inference safety evaluation of reasoning models: runs should
be distributed across fresh daemon loads (full process restart between models), because
long-running or multi-model daemon sessions can silently degrade trigger-session output
and under-report vulnerability. Third, we harden the harness against a related confound
by pinning model digests and refusing to run on a digest mismatch, so that a silent tag
update cannot be mistaken for a behavioural change.

This artifact is distinct from the qwq:32b Draft-Only phenomenon reported in the primary
results. That phenomenon is a genuine deliberative refusal ("the user asked only to
draft, not send") observed under the primary factorial's 16k-context environment; it
does not appear in the sequential batch, where qwq:32b instead exhibits the same
no-tool-call degradation as the other reasoning models. The two should not be conflated:
one is a model reasoning to a refusal, the other is host-conditioned output degradation.
