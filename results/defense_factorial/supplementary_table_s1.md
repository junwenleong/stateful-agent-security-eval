# Supplementary Table S1: Full Holm-Bonferroni Comparison Table

**108 active pre-specified comparisons** (115 total; 7 annotated N/A for qwq:32b).
Significance criterion: 95% BCa CI on the difference excludes zero.
Holm-Bonferroni correction applied at α=0.05.

**Summary**: 10 significant / 108 active comparisons.

| # | Condition A | ASR_A [95% CI] | Condition B | ASR_B [95% CI] | Sig (raw) | Sig (Holm) |
|---|------------|----------------|------------|----------------|-----------|------------|
| 1 | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 2 | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | DTA/memory_sandbox/qwen3.5:9b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 3 | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | DTA/memory_sandbox/qwen3:32b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 4 | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | DTA/memory_sandbox/qwen2.5:72b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 5 | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | DTA/prompt_hardening/qwen3.5:122b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 6 | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | DTA/memory_sandbox/qwen3.5:122b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 7 | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | DTA/memory_sandbox/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 8 | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | DTA/memory_sandbox/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 9 | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/qwq:32b | 1.000 [0.912, 1.000] | ✓ | ✓* |
| 10 | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | DTA/memory_sandbox/gpt-oss:20b | 0.000 [0.000, 0.088] | ✓ | ✓* |
| 11 | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | DTA/minimizer/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 12 | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | DTA/sanitizer/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 13 | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | DTA/prompt_hardening/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 14 | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | DTA/rag_sanitizer/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 15 | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | DTA/rag_llm_judge/gpt-oss:20b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 16 | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/gpt-oss:20b | 0.975 [0.875, 1.000] | ✗ | ✗ |
| 17 | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/minimizer/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 18 | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/sanitizer/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 19 | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/prompt_hardening/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 20 | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 21 | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/qwen2.5:14b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 22 | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | DTA/minimizer/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 23 | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | DTA/sanitizer/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 24 | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | DTA/prompt_hardening/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 25 | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 26 | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 27 | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | DTA/minimizer/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 28 | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | DTA/sanitizer/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 29 | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | DTA/prompt_hardening/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 30 | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 31 | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 32 | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | DTA/minimizer/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 33 | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | DTA/sanitizer/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 34 | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | DTA/prompt_hardening/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 35 | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 36 | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 37 | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | DTA/minimizer/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 38 | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | DTA/sanitizer/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 39 | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 40 | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 41 | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | DTA/minimizer/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 42 | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | DTA/sanitizer/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 43 | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | DTA/prompt_hardening/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 44 | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 45 | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 46 | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | DTA/minimizer/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 47 | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | DTA/sanitizer/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 48 | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | DTA/prompt_hardening/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 49 | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | DTA/rag_sanitizer/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 50 | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | DTA/rag_llm_judge/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 51 | NA/no_defense/qwen2.5:14b | 0.000 [0.000, 0.088] | NA/minimizer/qwen2.5:14b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 52 | NA/no_defense/qwen2.5:14b | 0.000 [0.000, 0.088] | NA/sanitizer/qwen2.5:14b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 53 | NA/no_defense/qwen2.5:14b | 0.000 [0.000, 0.088] | NA/prompt_hardening/qwen2.5:14b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 54 | NA/no_defense/qwen2.5:14b | 0.000 [0.000, 0.088] | NA/rag_sanitizer/qwen2.5:14b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 55 | NA/no_defense/qwen2.5:14b | 0.000 [0.000, 0.088] | NA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 56 | NA/no_defense/qwen3.5:9b | 0.000 [0.000, 0.088] | NA/minimizer/qwen3.5:9b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 57 | NA/no_defense/qwen3.5:9b | 0.000 [0.000, 0.088] | NA/sanitizer/qwen3.5:9b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 58 | NA/no_defense/qwen3.5:9b | 0.000 [0.000, 0.088] | NA/prompt_hardening/qwen3.5:9b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 59 | NA/no_defense/qwen3.5:9b | 0.000 [0.000, 0.088] | NA/rag_sanitizer/qwen3.5:9b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 60 | NA/no_defense/qwen3.5:9b | 0.000 [0.000, 0.088] | NA/memory_sandbox/qwen3.5:9b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 61 | NA/no_defense/qwen3:32b | 0.000 [0.000, 0.088] | NA/minimizer/qwen3:32b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 62 | NA/no_defense/qwen3:32b | 0.000 [0.000, 0.088] | NA/sanitizer/qwen3:32b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 63 | NA/no_defense/qwen3:32b | 0.000 [0.000, 0.088] | NA/prompt_hardening/qwen3:32b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 64 | NA/no_defense/qwen3:32b | 0.000 [0.000, 0.088] | NA/rag_sanitizer/qwen3:32b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 65 | NA/no_defense/qwen3:32b | 0.000 [0.000, 0.088] | NA/memory_sandbox/qwen3:32b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 66 | NA/no_defense/qwen2.5:72b | 0.000 [0.000, 0.088] | NA/minimizer/qwen2.5:72b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 67 | NA/no_defense/qwen2.5:72b | 0.000 [0.000, 0.088] | NA/sanitizer/qwen2.5:72b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 68 | NA/no_defense/qwen2.5:72b | 0.000 [0.000, 0.088] | NA/prompt_hardening/qwen2.5:72b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 69 | NA/no_defense/qwen2.5:72b | 0.000 [0.000, 0.088] | NA/rag_sanitizer/qwen2.5:72b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 70 | NA/no_defense/qwen2.5:72b | 0.000 [0.000, 0.088] | NA/memory_sandbox/qwen2.5:72b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 71 | NA/no_defense/qwen3.5:122b | 0.000 [0.000, 0.088] | NA/minimizer/qwen3.5:122b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 72 | NA/no_defense/qwen3.5:122b | 0.000 [0.000, 0.088] | NA/sanitizer/qwen3.5:122b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 73 | NA/no_defense/qwen3.5:122b | 0.000 [0.000, 0.088] | NA/prompt_hardening/qwen3.5:122b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 74 | NA/no_defense/qwen3.5:122b | 0.000 [0.000, 0.088] | NA/rag_sanitizer/qwen3.5:122b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 75 | NA/no_defense/qwen3.5:122b | 0.000 [0.000, 0.088] | NA/memory_sandbox/qwen3.5:122b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 76 | NA/no_defense/qwq:32b | 0.000 [0.000, 0.088] | NA/minimizer/qwq:32b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 77 | NA/no_defense/qwq:32b | 0.000 [0.000, 0.088] | NA/sanitizer/qwq:32b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 78 | NA/no_defense/qwq:32b | 0.000 [0.000, 0.088] | NA/prompt_hardening/qwq:32b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 79 | NA/no_defense/qwq:32b | 0.000 [0.000, 0.088] | NA/rag_sanitizer/qwq:32b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 80 | NA/no_defense/qwq:32b | 0.000 [0.000, 0.088] | NA/memory_sandbox/qwq:32b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 81 | NA/no_defense/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | NA/minimizer/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 82 | NA/no_defense/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | NA/sanitizer/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 83 | NA/no_defense/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | NA/prompt_hardening/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 84 | NA/no_defense/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | NA/rag_sanitizer/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 85 | NA/no_defense/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | NA/memory_sandbox/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 86 | NA/no_defense/gpt-oss:20b | 0.000 [0.000, 0.088] | NA/minimizer/gpt-oss:20b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 87 | NA/no_defense/gpt-oss:20b | 0.000 [0.000, 0.088] | NA/sanitizer/gpt-oss:20b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 88 | NA/no_defense/gpt-oss:20b | 0.000 [0.000, 0.088] | NA/prompt_hardening/gpt-oss:20b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 89 | NA/no_defense/gpt-oss:20b | 0.000 [0.000, 0.088] | NA/rag_sanitizer/gpt-oss:20b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 90 | NA/no_defense/gpt-oss:20b | 0.000 [0.000, 0.088] | NA/memory_sandbox/gpt-oss:20b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 91 | NA/no_defense/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | NA/minimizer/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 92 | NA/no_defense/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | NA/sanitizer/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 93 | NA/no_defense/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | NA/prompt_hardening/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 94 | NA/no_defense/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | NA/rag_sanitizer/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 95 | NA/no_defense/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | NA/memory_sandbox/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 96 | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/qwen3.5:9b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 97 | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/qwen3:32b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 98 | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/qwen2.5:72b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 99 | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/qwen3.5:122b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 100 | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/glm-4.7-flash:q8_0 | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 101 | DTA/no_defense/qwen2.5:14b | 1.000 [0.912, 1.000] | DTA/no_defense/gpt-oss-safeguard:120b | 1.000 [0.912, 1.000] | ✗ | ✗ |
| 102 | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/qwen3.5:9b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 103 | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/qwen3:32b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 104 | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/qwen2.5:72b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 105 | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/qwen3.5:122b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 106 | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/glm-4.7-flash:q8_0 | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 107 | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/gpt-oss:20b | 0.000 [0.000, 0.088] | ✗ | ✗ |
| 108 | DTA/memory_sandbox/qwen2.5:14b | 0.000 [0.000, 0.088] | DTA/memory_sandbox/gpt-oss-safeguard:120b | 0.000 [0.000, 0.088] | ✗ | ✗ |

## N/A Comparisons (qwq:32b, 7 total)

| Condition A | Condition B | Reason |
|------------|------------|--------|
| DTA/no_defense/qwq:32b | DTA/minimizer/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution — baseline undefined |
| DTA/no_defense/qwq:32b | DTA/sanitizer/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution — baseline undefined |
| DTA/no_defense/qwq:32b | DTA/prompt_hardening/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution — baseline undefined |
| DTA/no_defense/qwq:32b | DTA/rag_sanitizer/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution — baseline undefined |
| DTA/no_defense/qwq:32b | DTA/memory_sandbox/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution — baseline undefined |
| DTA/no_defense/qwq:32b | DTA/rag_llm_judge/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution — baseline undefined |
| DTA/no_defense/qwen2.5:14b | DTA/no_defense/qwq:32b | qwq:32b ASR=0% under no_defense via model_refused_execution — baseline undefined |