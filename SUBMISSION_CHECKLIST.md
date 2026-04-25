# arXiv Submission Checklist

## Title
Architectural Limits of Defense Against Persistent Memory Attacks on LLM Agents

## arXiv Metadata
- Primary: cs.CR (Cryptography and Security)
- Secondary: cs.LG (Machine Learning)  
- Tertiary: cs.AI (Artificial Intelligence)
- Comments: 9 models × 7 defenses × 2 attacks × N=40 = 5,040 runs; pre-registered comparisons; BCa bootstrap CIs throughout. Code and results: [GitHub URL]

## Pre-submission steps

### 1. Build clean submission Markdown
Run: `.venv/bin/python scripts/build_paper.py`
Output: `paper_submission.md`

### 2. Fix Unicode arrows for pdflatex compatibility
arXiv uses pdflatex by default. The `→` character requires `\usepackage[utf8]{inputenc}`.
Either:
  a) Add `--variable header-includes="\usepackage[utf8]{inputenc}"` to Pandoc command, OR
  b) Replace `→` with `->` in paper_submission.md before PDF generation:
     `sed -i 's/→/->/g' paper_submission.md`

### 3. Generate PDF
```bash
pandoc paper_submission.md -o paper.pdf \
  --pdf-engine=xelatex \
  --variable geometry:margin=1in \
  --variable fontsize=11pt \
  --variable header-includes="\usepackage[utf8]{inputenc}"
```
Or on arXiv's system (pdflatex):
```bash
pandoc paper_submission.md -o paper.tex
# Edit paper.tex to add \usepackage[utf8]{inputenc} in preamble
pdflatex paper.tex
```

### 4. Verify in PDF
- [ ] Archetype table renders (footnotes * and † visible)
- [ ] Wilson Score CIs display as [X, Y] not as references
- [ ] Backtick code spans render as inline code (monospace)
- [ ] Arrows render correctly (→ or ->)
- [ ] No overfull hbox warnings on long model names in table

### 5. Files to include in arXiv submission
- `paper_submission.md` (or converted .tex)
- `results/defense_factorial/supplementary_table_s1.md` (Supplementary Table S1)
- `canonical_numbers.md` (supplementary data)

### 6. GitHub repo
- README updated: 9 models, 5,040 runs ✅
- requirements.txt: all pinned ✅
- .env.example: AWS variables added ✅
- experiments/configs/defense_factorial.yaml: pre-registration artifact ✅
- results/defense_factorial/supplementary_table_s1.md: Holm-Bonferroni table ✅

## Known rendering issues
- `→` arrows: Unicode, needs inputenc for pdflatex. Fix with sed or xelatex.
- Archetype table is wide (5 columns) — may need `\small` font or landscape orientation
  if it overflows page width. Check in PDF.
