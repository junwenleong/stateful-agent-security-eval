# Supplementary Table S1: Full Holm-Bonferroni Comparison Table

**108 active pre-specified comparisons** (115 total; 7 annotated N/A for qwq:32b).
Significance criterion: Holm-Bonferroni step-down correction (alpha=0.05) applied to the
per-comparison Fisher exact p-value on each 2x2 (successes, failures) contingency table; a
comparison is significant when its Holm-corrected Fisher p is at most alpha. As a concordant
sign check we also record whether the 95% difference interval excludes zero (BCa when both arms
are non-degenerate, else the conservative Wilson-difference bound); the two criteria agree on all
108 comparisons. This matches `scripts/analyze_results.py` and `analysis.json`, which score the
no-attack comparisons on `btcr_success` and the attack comparisons on `attack_success`.

The **Metric** column states the quantity scored for each row: ASR (attack success rate,
attack arm) or BTCR (benign task completion rate, no-attack arm). The Rate_A / Rate_B columns
report the point estimate and 95% CI of that metric for each condition, so the displayed value
is the same quantity the significance flag tests. A trailing `*` on the Holm flag marks a
Holm-significant comparison.

**Summary**: 10 significant / 108 active comparisons.

| # | Metric | Condition A | Rate_A [95% CI] | Condition B | Rate_B [95% CI] | Sig (raw) | Sig (Holm) |
|---|--------|------------|-----------------|------------|-----------------|-----------|------------|
| 1 | ASR | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 2 | ASR | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | DTA/memory_sandbox/qwen3.5:9b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 3 | ASR | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | DTA/memory_sandbox/qwen3:32b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 4 | ASR | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | DTA/memory_sandbox/qwen2.5:72b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 5 | ASR | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | DTA/prompt_hardening/qwen3.5:122b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 6 | ASR | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | DTA/memory_sandbox/qwen3.5:122b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 7 | ASR | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | DTA/memory_sandbox/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 8 | ASR | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | DTA/memory_sandbox/gpt-oss:20b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 9 | ASR | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | DTA/memory_sandbox/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 10 | ASR | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/qwq:32b | 1.000 [0.912, 1.000] | ✓ | ✓* |
| 11 | ASR | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/minimizer/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 12 | ASR | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/sanitizer/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 13 | ASR | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/prompt_hardening/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 14 | ASR | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 15 | ASR | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 16 | ASR | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | DTA/minimizer/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 17 | ASR | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | DTA/sanitizer/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 18 | ASR | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | DTA/prompt_hardening/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 19 | ASR | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 20 | ASR | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 21 | ASR | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | DTA/minimizer/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 22 | ASR | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | DTA/sanitizer/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 23 | ASR | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | DTA/prompt_hardening/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 24 | ASR | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 25 | ASR | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 26 | ASR | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | DTA/minimizer/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 27 | ASR | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | DTA/sanitizer/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 28 | ASR | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | DTA/prompt_hardening/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 29 | ASR | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 30 | ASR | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 31 | ASR | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | DTA/minimizer/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 32 | ASR | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | DTA/sanitizer/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 33 | ASR | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 34 | ASR | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 35 | ASR | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | DTA/minimizer/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 36 | ASR | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | DTA/sanitizer/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 37 | ASR | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | DTA/prompt_hardening/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 38 | ASR | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 39 | ASR | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 40 | ASR | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | DTA/minimizer/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 41 | ASR | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | DTA/sanitizer/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 42 | ASR | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | DTA/prompt_hardening/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 43 | ASR | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | DTA/rag_sanitizer/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 44 | ASR | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | DTA/rag_llm_judge/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 45 | ASR | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | DTA/minimizer/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 46 | ASR | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | DTA/sanitizer/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 47 | ASR | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | DTA/prompt_hardening/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 48 | ASR | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 49 | ASR | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 50 | BTCR | NA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | NA/minimizer/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 51 | BTCR | NA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | NA/sanitizer/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 52 | BTCR | NA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | NA/prompt_hardening/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 53 | BTCR | NA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | NA/rag_sanitizer/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 54 | BTCR | NA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | NA/memory_sandbox/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 55 | BTCR | NA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | NA/minimizer/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 56 | BTCR | NA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | NA/sanitizer/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 57 | BTCR | NA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | NA/prompt_hardening/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 58 | BTCR | NA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | NA/rag_sanitizer/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 59 | BTCR | NA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | NA/memory_sandbox/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 60 | BTCR | NA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | NA/minimizer/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 61 | BTCR | NA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | NA/sanitizer/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 62 | BTCR | NA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | NA/prompt_hardening/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 63 | BTCR | NA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | NA/rag_sanitizer/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 64 | BTCR | NA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | NA/memory_sandbox/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 65 | BTCR | NA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | NA/minimizer/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 66 | BTCR | NA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | NA/sanitizer/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 67 | BTCR | NA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | NA/prompt_hardening/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 68 | BTCR | NA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | NA/rag_sanitizer/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 69 | BTCR | NA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | NA/memory_sandbox/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 70 | BTCR | NA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | NA/minimizer/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 71 | BTCR | NA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | NA/sanitizer/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 72 | BTCR | NA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | NA/prompt_hardening/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 73 | BTCR | NA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | NA/rag_sanitizer/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 74 | BTCR | NA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | NA/memory_sandbox/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 75 | BTCR | NA/no_defense/qwq:32b | 1.000 [0.912, 1.000] | NA/minimizer/qwq:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 76 | BTCR | NA/no_defense/qwq:32b | 1.000 [0.912, 1.000] | NA/sanitizer/qwq:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 77 | BTCR | NA/no_defense/qwq:32b | 1.000 [0.912, 1.000] | NA/prompt_hardening/qwq:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 78 | BTCR | NA/no_defense/qwq:32b | 1.000 [0.912, 1.000] | NA/rag_sanitizer/qwq:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 79 | BTCR | NA/no_defense/qwq:32b | 1.000 [0.912, 1.000] | NA/memory_sandbox/qwq:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 80 | BTCR | NA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | NA/minimizer/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 81 | BTCR | NA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | NA/sanitizer/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 82 | BTCR | NA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | NA/prompt_hardening/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 83 | BTCR | NA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | NA/rag_sanitizer/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 84 | BTCR | NA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | NA/memory_sandbox/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 85 | BTCR | NA/no_defense/gpt-oss:20b | 1.000 [0.912, 1.000] | NA/minimizer/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 86 | BTCR | NA/no_defense/gpt-oss:20b | 1.000 [0.912, 1.000] | NA/sanitizer/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 87 | BTCR | NA/no_defense/gpt-oss:20b | 1.000 [0.912, 1.000] | NA/prompt_hardening/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 88 | BTCR | NA/no_defense/gpt-oss:20b | 1.000 [0.912, 1.000] | NA/rag_sanitizer/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 89 | BTCR | NA/no_defense/gpt-oss:20b | 1.000 [0.912, 1.000] | NA/memory_sandbox/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 90 | BTCR | NA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | NA/minimizer/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 91 | BTCR | NA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | NA/sanitizer/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 92 | BTCR | NA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | NA/prompt_hardening/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 93 | BTCR | NA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | NA/rag_sanitizer/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 94 | BTCR | NA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | NA/memory_sandbox/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 95 | ASR | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 96 | ASR | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 97 | ASR | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 98 | ASR | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 99 | ASR | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 100 | ASR | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | ✗ | ✗ |
| 101 | ASR | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 102 | ASR | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/qwen3.5:9b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 103 | ASR | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/qwen3:32b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 104 | ASR | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/qwen2.5:72b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 105 | ASR | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/qwen3.5:122b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 106 | ASR | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 107 | ASR | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/gpt-oss:20b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 108 | ASR | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | ✗ | ✗ |

## N/A Comparisons (qwq:32b, 7 total)

| Condition A | Condition B | Reason |
|------------|------------|--------|
| DTA/no_defense/qwq:32b | DTA/minimizer/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution; baseline undefined |
| DTA/no_defense/qwq:32b | DTA/sanitizer/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution; baseline undefined |
| DTA/no_defense/qwq:32b | DTA/prompt_hardening/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution; baseline undefined |
| DTA/no_defense/qwq:32b | DTA/rag_sanitizer/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution; baseline undefined |
| DTA/no_defense/qwq:32b | DTA/memory_sandbox/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution; baseline undefined |
| DTA/no_defense/qwq:32b | DTA/rag_llm_judge/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution; baseline undefined |
| DTA/no_defense/qwen2.5:14b | DTA/no_defense/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution; baseline undefined |
