# arXiv Submission Checklist

> **Last updated**: 2026-06-27. Source: `paper/paper.tex`

## Current Status

**Paper is complete and submittable as-is.** All experiments on this machine are done. Waiting on Mac Studio sprint (~4 days, launched 2026-06-26 18:00 SGT):

| Phase | What | ETA | Impact on paper |
|---|---|---|---|
| 0 | Cross-model date sweep (4 models × 2 dates × N=10) | ~5h | Minor footnote in §4.6 |
| 1 | Payload variants (5 framings × qwen2.5:14b × N=10) | ~1h | New paragraph §3.1 if ASR stays 100% |
| 2 | RATG factorial (9 models × 2 defenses × 2 attacks × N=40) | ~3 days | **Major** — new §3.3.x if ASR drops |
| 3 | 7B judge (3 models × 2 defenses × N=40) | ~8h | Revise §3.2.4 if detection improves |

All sprint results are **additive** — nothing in the current draft needs rewriting regardless of outcomes.

**Completed experiments (this machine):**
- ✅ frontier API frontier probe: 21 models × N=10 = 210 runs, 0% ASR
- ✅ frontier API sandbox probe: 4 Latent Carriers × N=10-16 = 46 runs, 0% bypass
- ✅ Bedrock N=40 date sweep: 5 models × 3 dates, all p>0.017 (null confirmed)
- ✅ Supply chain: parked (logical argument in paper)
- ✅ Defense factorial: 5,040 runs (Mac Studio, completed April)
- ✅ N=10 rescreen: 180 runs (Mac Studio, completed April)
- ✅ Bedrock Sonnet/Haiku N=100: 400 runs (completed April)

## Metadata
- **Title**: Defense Effectiveness Across Architectural Layers: A Mechanistic Evaluation of Persistent Memory Attacks on Stateful LLM Agents
- **Primary**: cs.CR (Cryptography and Security)
- **Secondary**: cs.LG, cs.AI
- **Comments**: 5,040 factorial runs + 210 frontier screening runs + 46 sandbox probe runs; 9 open-source models, 21 frontier models, 3 providers; pre-registered comparisons; BCa bootstrap CIs.

## Build

```bash
cd paper/
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Or package for arXiv:
```bash
tar -czf paper-v4-arxiv.tar.gz paper.tex math_commands.tex references.bib iclr2025_conference.sty paper.bbl
```

## Pre-submission checks

- [ ] `paper/paper.tex` compiles without errors
- [ ] All `\ref{sec:*}` resolve (no `??`)
- [ ] Abstract ≤ 1500 chars (arXiv limit for metadata field)
- [ ] All numbers match `canonical_numbers.md`
- [ ] Supplementary Table S1 (Holm-Bonferroni) referenced
- [ ] Frontier screening table renders correctly (21 models)
- [ ] No frontier API/infrastructure details in paper text
- [ ] Author name and email correct

## Source of truth hierarchy

1. `paper/paper.tex` — submission text
2. `canonical_numbers.md` — every number traces here
3. `FINDINGS.md` + `docs/index.md` — public reader-facing
4. `README.md` — repo landing page

## GitHub repo (public-facing)

- [x] README: 5,040 runs, 9 models, 21 frontier
- [x] FINDINGS.md: complete results summary
- [x] docs/index.md: GitHub Pages site (v4 section)
- [x] canonical_numbers.md: experiments 1-6
- [x] scripts/verify_canonical.py: programmatic verification
