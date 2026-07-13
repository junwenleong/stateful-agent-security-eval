
---

## NDSS 2027 Fall Cycle Submission

> **Target deadline**: Aug 19, 2026. Notification: Nov 4, 2026. Seoul, March 22-26 2027.
> **Track**: Security and privacy of systems based on ML/AI/LLMs
> **Paper**: `paper/ndss/paper.tex` (IEEEtran two-column, 12 pages body)

### Status (updated 2026-07-11)

- [x] Paper skeleton and section structure
- [x] Full paper written (544 lines, 12 pages, compiling clean)
- [x] Abstract compressed to ~200 words
- [x] Threat Model section (new for NDSS)
- [x] All key numbers from claims_ledger_ndss.md present
- [x] verify_canonical.py: ALL CHECKS PASSED
- [x] Double-blind: author anonymized, no GitHub URLs, no self-identifying references
- [x] PersistBench artifact built (22 files, smoke test offline <1s)
- [x] Claims ledger locked (384 lines, 10 claims, 9-paper prior-art matrix)
- [ ] Artifact packaging (PersistBench + raw data + reproduce scripts)
- [ ] Internal adversarial red-team review
- [ ] Final anonymization audit (PDF metadata, git history, package names)
- [ ] Submit

### Build

```bash
cd paper/ndss/
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

### Artifact (targeting all 3 badges: Available, Functional, Reproduced)

- PersistBench v1.0.0 at `persistbench/`
- Smoke test: `persistbench run --profile smoke` (offline, <1s)
- Core test: `persistbench run --profile core` (representative subset, API keys needed)
- Full: `persistbench run --profile full` (all 5,040 runs)
- Reproduce paper figures: `make -C persistbench figures`

### Key numbers (must match paper)

- 5,040 pre-registered factorial runs, zero errors
- 88.9% overall ASR on failing defenses (320/360)
- Memory Sandbox: 0% ASR on 8/9 models
- qwq:32b reasoning bypass: inverts Memory Sandbox to 100%
- Tripartite vendor: Anthropic 0% injection, OpenAI generational hardening, Google 95% ASR
- 9 open-source + 21 frontier models, 3 vendors
# arXiv Submission Checklist

> **Last updated**: 2026-07-08. Source: `paper/paper.tex`

## Current Status

**Paper v4 submitted to arXiv on 2026-07-03.** v3 superseded.

**What's done (clean, publishable RATG evidence, 4 cells):**
- Phase 0 (date sweep): ✅ complete
- Phase 1 (payload variants): ✅ complete (50/50)
- 3 mechanical models RATG: ✅ clean (100%→0%, qwen2.5:14b/72b, qwen3:32b)
- 1 reasoning model RATG: ✅ clean (100%→0%, qwen3.5:122b, arms session-0-identical), Iteration 59

**Fresh-load reasoning models NOT usable as RATG evidence (Iteration 59, per-model reason):**
- qwq:32b: session-0 arm fork (RATG inert until recall; arm contrast not a defense effect)
- gpt-oss:20b (80/80): session-0 arm fork (no_defense saves 6 in S0, RATG saves 1); do NOT read the 100%→2.5% as a defense effect
- qwen3.5:9b: runtime-version misparse, both arms 0%, no vulnerable baseline
- gpt-oss-safeguard:120b (22/40, no_defense only): Draft-Only shift on Ollama 0.30.11 (~4.5% no_defense ASR vs factorial 100%), no vulnerable baseline. Runtime/environment-dependent, NOT proven version-dependent without a 0.20.6/0.30.11 A/B.

**COMPLETE (Iteration 61, 2026-07-03):**
- ✅ glm-4.7-flash:bf16 fresh-load RATG: S0-fork, uninterpretable (Iteration 60)
- ✅ 7B-judge suite: VALID (0/120 malicious-doc flagged, 0/400 total, ASR 100%→100% on qwen2.5:14b + qwen3:32b); 122b uninterpretable (baseline degraded)
- ✅ 122b arm-order control: falsified 100/0/0/100 prediction; led to full reproducibility investigation
- ✅ 122b RATG DOWNGRADED (baseline non-reproducible post-reboot, Iteration 61)

**v4 writing actions (DONE, uncommitted in working tree):**
1. ✅ Daemon-degradation appendix updated: GUI-supervisor/effective-config note, GPU-pressure ruled out, "Fresh-load retests and baseline non-reproducibility" paragraph
2. ✅ RATG section rewritten: 3 mechanical only, two-part reproducibility validity criterion, 122b excluded, no reasoning-model efficacy claim
3. ✅ qwq + gpt-oss:20b + 122b/9b/glm contrast documented per model with specific failure reason
4. ✅ 7B-judge result slotted into §3.2.4 (capability-bound: scaling 1.5B→7B did not detect this payload)
5. ✅ Methods/reproducibility: effective-config invariant added
6. ✅ verify_canonical.py: verify_ratg() + verify_judge7b() pass
7. Do NOT add BSI §4.7/§4.8 reasoning-class instability claim (retracted, Iteration 55)
8. Do NOT widen "qwq-specific" scoping (sprint 0% was degradation, not safety)

**When sprint Phase 3 (7B judge) completes:**
- If 7B detects what 1.5B couldn't: revise §3.2.4 to bound the capability threshold. "A 7B judge achieves X% detection vs 0% for 1.5B."
- If 7B also fails: strengthen the "semantic masking defeats small judges" narrative. Add 1 sentence.

**Temporal persistence (already added to paper.tex §3.1):**
- ✅ Marathon test cited: "50-session marathon, byte-identical at checkpoints 10/20/30/40/50, persistence bounded by database durability."

**Completed experiments (this machine):**
- ✅ frontier model probe: 21 models × N=10 = 210 runs, 0% ASR
- ✅ frontier sandbox probe: 4 Latent Carriers × N=10-16 = 46 runs, 0% bypass
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

1. `paper/paper.tex` - submission text
2. `canonical_numbers.md` - every number traces here
3. `FINDINGS.md` + `docs/index.md` - public reader-facing
4. `README.md` - repo landing page

## GitHub repo (public-facing)

- [x] README: 5,040 runs, 9 models, 21 frontier
- [x] FINDINGS.md: complete results summary
- [x] docs/index.md: GitHub Pages site (v4 section)
- [x] canonical_numbers.md: experiments 1-6
- [x] scripts/verify_canonical.py: programmatic verification
