# arXiv Submission Checklist

> **Last updated**: 2026-06-27. Source: `paper/paper.tex`

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
