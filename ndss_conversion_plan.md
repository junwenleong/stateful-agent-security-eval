# NDSS 2027 Conversion Plan

> arXiv:2605.08442 (v4, ICLR format) --> NDSS 2027 Submission
> Target: "Security and privacy of systems based on ML/AI/LLMs" track
> Deadline: TBD (typically April for Summer, September for Fall cycle)

## 1. Format Changes

### Template Swap

| Property | Current (ICLR 2025) | Target (NDSS 2027) |
|----------|---------------------|---------------------|
| Class | `\documentclass{article}` + `iclr2025_conference.sty` | `\documentclass[conference]{ndss}` (NDSS LaTeX template) |
| Columns | Single-column | Two-column |
| Paper size | US Letter (ICLR) | US Letter |
| Font | Times 10pt (ICLR) | Times New Roman 10pt, 11pt leading |
| Page limit | No hard limit (was ~18 pages + appendix) | 13 pages (excl. refs, ethics, appendices) |
| Margins | ICLR default (~1in) | NDSS template default |
| References | `natbib` numeric | IEEE/NDSS style (numeric brackets) |
| Abstract | Unlimited (currently ~500 words) | Typically 150-250 words for NDSS |

### Files to Replace/Add

- **Remove:** `iclr2025_conference.sty`, `iclr2025_conference.bst`, `fancyhdr.sty`, `natbib.sty`
- **Add:** `ndss2027.cls` (or whichever NDSS provides), NDSS `.bst` file
- **Modify:** `paper.tex` header, bibliography style, figure/table sizing for two-column

### Immediate Formatting Actions

1. Replace `\documentclass{article}` + `\usepackage{iclr2025_conference,times}` with NDSS class
2. Remove `\iclrfinalcopy` (NDSS uses anonymous submission by default)
3. Remove `\lhead{Preprint. Under review.}`
4. Resize all tables for two-column width (most will need `table*` for full-width or significant compression)
5. Convert the TikZ pipeline figure to fit single-column or span two columns
6. Adjust `\footnotesize`/`\small` table fonts for readability at two-column width

---

## 2. Anonymization Requirements

### Must Remove

| Item | Location | Action |
|------|----------|--------|
| Author name | `\author{Jun Wen Leong}` | Replace with anonymous submission boilerplate |
| Email | `\texttt{leongjunwen@gmail.com}` | Remove entirely |
| GitHub URL | `\url{https://github.com/junwenleong/stateful-agent-security-eval}` | Replace with "available upon publication" or anonymous link |
| Self-citation | `\citet{leong2025forensic}` (companion work) | Convert to "prior work [Anonymous]" or "Anonymous (2025)" with note |
| arXiv preprint markers | Header text | Remove all preprint indicators |

### Self-Citation Strategy

The paper cites `\citet{leong2025forensic}` (arXiv:2606.30566, the forensic trajectory signatures paper). Options:
1. **Preferred:** Cite as "[Anonymous, 2025]" with footnote "Details omitted for blind review; will be restored in camera-ready"
2. **Alternative:** Cite normally but in third person ("Leong (2025) showed...") -- NDSS typically requires option 1

### Anonymization Checklist

- [ ] No author names anywhere in PDF (including metadata)
- [ ] No institutional affiliation
- [ ] No acknowledgments section (or anonymized)
- [ ] Self-citations in third person or anonymized
- [ ] No GitHub/repo URLs that identify the author
- [ ] PDF metadata scrubbed (no author field in `\hypersetup`)
- [ ] No "our prior work" or "in our companion paper" phrasing that de-anonymizes

---

## 3. Section-by-Section Mapping

### Current Structure (ICLR, ~18 pages body + 4 pages appendix)

| Section | Current Pages (est.) | Content |
|---------|---------------------|---------|
| Abstract | ~1 page | 500+ words, very detailed |
| 1. Introduction | ~1.5 pages | Threat model, contributions (5 items) |
| 2. Methodology | ~3 pages | Attack, defenses, mechanistic analysis, design, stats, models |
| 3. Results | ~6 pages | Attack char., 5 failures, sandbox, reasoning-defense, RATG, artifacts, frontier |
| 4. Frontier Generational | ~3 pages | Empty corpus, Gemini, OpenAI, vendor divergence, system prompt, supply chain, date |
| 5. Discussion | ~3 pages | Arch layer, tool-gating, persistence, env variable, implications, frontier impl., supply chain |
| 6. Related Work | ~1.5 pages | |
| 7. Conclusion | ~0.5 page | |
| Ethics | ~0.3 page | |
| Appendix A (Artifacts) | ~0.5 page | |
| Appendix B (Daemon) | ~2 pages | |
| Appendix C (Bedrock) | ~1 page | |
| Appendix D (Provenance) | ~0.5 page | |

**Estimated current body:** ~18-20 pages single-column ≈ 10-11 pages two-column (two-column is ~60% of single-column page count for text-heavy papers)

### Proposed NDSS Structure (13 pages two-column)

| # | NDSS Section | Source | Est. Pages | Notes |
|---|---|---|---|---|
| | **Abstract** | Rewrite (compress to 200 words) | 0.15 | |
| 1 | **Introduction** | Current S1, reframed for security audience | 1.5 | Sharpen threat model, emphasize practical impact |
| 2 | **Background and Threat Model** | NEW (partially from S1 + S2.1) | 1.0 | NDSS expects explicit threat model section |
| 3 | **System Design** | Current S2.2-2.3 (defenses, mechanistic) | 1.5 | Evaluation framework as a "system" |
| 4 | **Experimental Setup** | Current S2.4-2.6 (design, stats, models) | 1.0 | Compress; move details to appendix |
| 5 | **Evaluation Results** | Current S3.1-3.3 (attack char, failures, sandbox) | 2.5 | Core contribution; keep most detail |
| 6 | **Reasoning-Defense Interaction** | Current S3.4-3.5 (double dissociation, RATG) | 2.0 | Key novelty; keep |
| 7 | **Frontier Model Evaluation** | Current S4 (generational, vendor divergence) | 2.0 | Compress date sweep + supply chain |
| 8 | **Discussion** | Current S5 (selected subsections) | 1.0 | Merge deployment implications |
| 9 | **Related Work** | Current S6 + new 2026 prior art | 1.0 | Expand with recent work |
| 10 | **Conclusion** | Current S7 | 0.3 | |
| | **Appendices** (not counted) | Current App A-D + new additions | ~4 | |

**Total body: ~13.0 pages** (tight but feasible)

---

## 4. Content Expansion Opportunities

We have ~13 pages to fill in two-column. The current paper in two-column would be ~10-11 pages. We need to ADD approximately 2-3 pages of new content.

### Sections to Expand

| Section | What to Add | Estimated Addition |
|---------|-------------|-------------------|
| **Background/Threat Model** | Formal threat model (attacker capabilities, assumptions, goals), system model diagram | +1.0 page |
| **Related Work** | 2026 prior art (see below), positioning against concurrent work | +0.5 page |
| **Ethics Statement** | Expand to full ethics section (NDSS requires this) | +0.3 page |
| **Artifact Description** | NDSS encourages artifact appendix (badges) | +0.3 page |
| **Discussion: Defenses** | Expand constructive defense discussion (SMSR, MemLineage, provenance) | +0.5 page |
| **Evaluation: Extended Analysis** | Any new results between now and submission | +0.5 page |

### New Prior Art to Incorporate (2026 papers)

These MUST be cited per the research.md prior-art requirements:

| Paper | Relevance | Section |
|-------|-----------|---------|
| **SMSR** (Sharma, arXiv:2606.12703) | HMAC-SHA256 provenance for agent memory; directly relevant defense | Related Work + Discussion |
| **Inseparability** (Pant et al., 2606.27567) | Formal impossibility of in-pipeline prompt injection defense | Related Work |
| **Surface Heuristics** (Li et al., ACL 2026, 2601.07185) | Trained defenses learn shortcuts, not intent | Related Work |
| **Role Confusion** (Ye et al., ICML 2026, 2603.12277) | Models infer source from style/content | Related Work |
| **MemLineage** (arXiv 2605.14421) | Per-principal Ed25519, Merkle logs | Related Work + Discussion |
| **PACT** (arXiv 2605.11039) | Argument-level provenance for tool calls | Discussion |
| **VectorPin/VectorSmuggle** (arXiv 2605.13764) | Ed25519 provenance for embeddings | Related Work |
| **TMA-NM** (arXiv 2606.24322) | Laundering through agent summaries | Related Work |
| **eTAMP** (arXiv 2604.02623) | Environment-injected cross-session memory poisoning | Related Work |
| **MPBench** (arXiv 2606.04329) | Memory poisoning benchmark | Related Work |
| **Sleeper Memory Poisoning** (arXiv 2605.15338) | Sleeper agent variant | Related Work |
| **AgentLAB** (arXiv 2602.16901) | Long-horizon attacks including memory | Related Work |
| **ASIDE** (arXiv 2503.10566) | Architectural instruction/data separation | Discussion |
| **Formalizing LLM Agent Security** (arXiv 2603.19469) | Formal framework | Related Work |
| **Adaptive Attacks Break Defenses** (arXiv 2503.00061) | Adaptive attacks defeat static defenses | Related Work |

### Content to Compress or Move to Appendix

| Current Content | Action | Reason |
|-----------------|--------|--------|
| Detailed RATG validity criterion per-model | Move most to appendix | Too granular for main text |
| Date sensitivity section (S4.6) | Compress to 1 paragraph in Discussion | Negative result, not core |
| Evaluation artifacts (S3.6) | Move to appendix entirely | Supporting detail |
| Full provenance table (App D) | Keep as appendix | Already there |
| Daemon degradation details (App B) | Keep as appendix, compress | Already there |
| qwq:32b temporal stability paragraph | Compress; key point in 2 sentences | Currently too long |
| Hallucination bypass paragraph | Compress to 2-3 sentences | Minor finding |

---

## 5. Structural Reframing for NDSS Audience

### Key Differences: ML Conference vs Security Conference

| Aspect | ICLR/ML Framing | NDSS/Security Framing |
|--------|-----------------|----------------------|
| Contribution type | "Mechanistic evaluation" | "Security evaluation of deployed systems" |
| Emphasis | Model behavior taxonomy | Attack surface analysis + defense gaps |
| Threat model | Implicit/scattered | Explicit section, formal attacker model |
| Defense evaluation | "What fails" | "Why current defenses are insufficient" + constructive path |
| Practical impact | Secondary | Primary (deployment recommendations) |
| Reproducibility | Methodology contribution | Artifact description + badges |
| Terminology | "Behavioral archetypes" | "Vulnerability classes" |

### Reframing the Title

Current: "Defense Effectiveness Across Architectural Layers: A Mechanistic Evaluation of Persistent Memory Attacks on Stateful LLM Agents"

Options for NDSS:
1. "Persistent Memory Attacks on LLM Agents: Why Five Defense Classes Fail and What Works" (direct, action-oriented)
2. "The Memory Layer Gap: Evaluating Defense Effectiveness Against Cross-Session Attacks on LLM Agents" (emphasizes the gap)
3. "When Compliance Becomes an Attack: Cross-Session Memory Poisoning in Agentic LLM Systems" (emphasizes the novel attack vector)

**Recommendation:** Option 1 -- NDSS reviewers value directness and clear security contribution.

### Reframing the Abstract

Compress from ~500 words to ~200 words. Focus on:
1. The attack (1-2 sentences): persistent memory + delayed trigger
2. The evaluation (1 sentence): 5,040 runs, 9 models, 6 defenses
3. The key finding (2-3 sentences): 5/6 defenses fail due to architectural placement; only tool-gating works but is bypassable by reasoning models
4. Frontier finding (1 sentence): 21 models, vendor divergence, Gemini 95% ASR
5. Implication (1 sentence): defense investment should target memory-action pathway

---

## 6. New Sections Required by NDSS

### 6.1 Explicit Threat Model Section (NEW)

Must contain:
- **System model:** LLM agent with persistent memory, RAG retrieval, tool-calling capability
- **Attacker model:** Can inject documents into RAG corpus (e.g., via shared document store, email attachment, web page). Cannot modify system prompt, tool implementations, or model weights.
- **Attacker goal:** Exfiltrate sensitive data via tool-calling (email forwarding)
- **Attacker capabilities:** Single injection opportunity; payload must survive across sessions
- **Assumptions:** Agent uses standard tool-calling interface; persistent memory is writable via tools; RAG corpus is partially attacker-controlled
- **Out of scope:** Adaptive attacks, model weight modification, multi-agent coordination (discuss as future work)

### 6.2 Ethics Statement (Expand)

Current ethics section is adequate but should be expanded for NDSS:
- No human subjects
- Synthetic data only (no real emails/documents)
- Responsible disclosure timeline (when disclosed, to whom, response received?)
- Dual-use consideration: attack methodology could be replicated
- Why we publish: defense community needs to understand the gap

### 6.3 Artifact Description (NEW appendix)

NDSS has an artifact evaluation process. Include:
- What artifacts are available (code, data, logs)
- How to reproduce key results
- Hardware requirements
- Expected runtime
- Badges sought (Available, Functional, Reproduced)

---

## 7. Anonymization Action Items

### Code Changes

```latex
% REMOVE:
\author{Jun Wen Leong \\ \texttt{leongjunwen@gmail.com}}

% REPLACE WITH:
\author{Anonymous Submission}
% or whatever NDSS template requires

% REMOVE:
\url{https://github.com/junwenleong/stateful-agent-security-eval}

% REPLACE WITH:
"The evaluation harness and full logs will be released upon publication."
% OR use anonymous hosting (e.g., anonymous.4open.science)

% SELF-CITATION:
% Current:
\citet{leong2025forensic}
% Replace with:
[Anonymous]~\cite{anon2025forensic}
% Add to references.bib:
@misc{anon2025forensic,
  author = {Anonymous},
  title = {[Title omitted for blind review]},
  year = {2025},
  note = {Under review. Details provided in supplementary material.}
}
```

### PDF Metadata

```latex
\hypersetup{
  pdfauthor={},  % EMPTY for blind review
  pdftitle={...},
  pdfsubject={},
  pdfkeywords={}
}
```

---

## 8. Implementation Plan (Ordered Steps)

### Phase 1: Template Swap (Day 1)

1. Download NDSS 2027 LaTeX template
2. Create `paper_ndss/` directory alongside existing `paper/`
3. Port content section-by-section into new template
4. Verify compilation
5. Check page count (target: body fits in ~11 pages to leave room for expansion)

### Phase 2: Restructure (Days 2-3)

1. Write new "Background and Threat Model" section
2. Merge current S4 (Frontier Generational) into main Results or separate S7
3. Compress/move: date sensitivity, artifacts, daemon appendix details
4. Split current Discussion into tighter subsections
5. Rewrite abstract (200 words max)
6. Rewrite introduction for security audience

### Phase 3: Expand (Days 4-5)

1. Add all 2026 prior art to Related Work
2. Expand Discussion with constructive defense directions (SMSR, MemLineage, provenance)
3. Write explicit threat model section
4. Expand ethics statement
5. Write artifact description appendix
6. Add any new experimental results available by submission time

### Phase 4: Anonymize (Day 6)

1. Remove all identifying information
2. Convert self-citations
3. Anonymize GitHub URLs
4. Scrub PDF metadata
5. Create anonymous artifact repository (anonymous.4open.science or similar)

### Phase 5: Polish (Days 7-8)

1. Verify all tables fit two-column format
2. Check figure rendering at smaller column width
3. Verify page count (exactly 13 or fewer)
4. Proofread for NDSS style consistency
5. Check all `\ref{}` resolve
6. Final compilation + PDF check

---

## 9. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Page overflow (>13 pages) | Medium | Pre-identify content for appendix; tables are the biggest space consumer |
| Tables too wide for two-column | High | Will need `table*` (full-width) for most tables or aggressive reformatting |
| Self-citation detectable | Medium | Use completely anonymous reference; avoid distinctive methodology descriptions |
| arXiv version findable | High | Acceptable for NDSS (they allow arXiv preprints); note in submission form |
| Reviewer overlap with ICLR | Low | Different community; NDSS reviewers are security-focused |
| Missing NDSS template | Low | Template usually available months before deadline; use prior year's if needed |

---

## 10. Key Decisions Needed

1. **Title:** Keep current or reframe for security audience? (Recommend reframe)
2. **arXiv:** NDSS allows preprints on arXiv. Do we update arXiv to NDSS format, or keep ICLR format on arXiv and submit separately? (Recommend: keep arXiv as-is, submit NDSS separately)
3. **New results:** Any experiments to run between now and submission that strengthen the NDSS version? (Suggest: adaptive attacker evaluation, or SMSR-style defense comparison)
4. **Companion paper citation:** How to handle the forensic trajectory paper (arXiv:2606.30566) under blind review?
5. **Artifact evaluation:** Submit for NDSS artifact badges? (Recommend yes -- strengthens acceptance)

---

## 11. Estimated Effort

| Phase | Time | Dependencies |
|-------|------|-------------|
| Template swap | 4 hours | NDSS template available |
| Restructure | 8 hours | None |
| Expand (new content) | 12 hours | Prior art reading |
| Anonymize | 2 hours | None |
| Polish + tables | 6 hours | None |
| **Total** | **~32 hours** | |

---

## 12. NDSS-Specific Strengths of This Paper

Why this paper is a good fit for NDSS:

1. **Practical security evaluation** -- not just ML behavioral study, but "here's what breaks in deployed systems"
2. **Defense-focused** -- NDSS values "what can defenders do?" over "here's a new attack"
3. **Multi-vendor evaluation** -- covers real production models (OpenAI, Google, Anthropic)
4. **Responsible disclosure** -- already done, demonstrates maturity
5. **Constructive path** -- RATG proof-of-concept + Memory Sandbox analysis gives defenders actionable guidance
6. **Reproducibility** -- full artifact release planned, aligns with NDSS artifact evaluation
7. **Timely** -- LLM agents with persistent memory are actively being deployed; the security gap is urgent
