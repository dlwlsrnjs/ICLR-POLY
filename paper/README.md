# PolyJigsaw — ICLR 2026 submission

Built on the official ICLR 2026 template (files fetched from ICLR/Master-Template):
`iclr2026_conference.{sty,bst}`, `math_commands.tex`, `fancyhdr.sty`, `natbib.sty`.

## Files
- `polyjigsaw_iclr2026.tex` — main paper (`\documentclass{article}` +
  `\usepackage{iclr2026_conference,times}`, per the template).
- `references.bib` — bibliography.
- `tab_curve.tex`, `tab_main.tex`, `tab_attaq.tex` — result tables, auto-generated
  from `results/*.json` by `scripts/make_paper_tables.py`. Re-run that script after
  the full-scale head-to-head (`run_paper_main.sh`) finishes to replace the
  `\pending` cells (CSRT, MD-Judge, HR, multi-target) with final numbers.

## Compile
```bash
pdflatex polyjigsaw_iclr2026
bibtex   polyjigsaw_iclr2026
pdflatex polyjigsaw_iclr2026
pdflatex polyjigsaw_iclr2026
```
(No LaTeX on this machine; compiles on Overleaf or any TeX Live install.)

## Regenerate tables
```bash
python3 ../scripts/make_paper_tables.py
```
