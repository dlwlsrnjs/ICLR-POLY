# PolyJigsaw manuscript revision

September 2026 review draft using the ICLR 2026 template. This is not a claim that an
active conference submission exists or that current submission requirements are met.

## Read the revision

- `build/polyjigsaw_iclr2026.pdf`: rendered paper, references, and appendices.
- `polyjigsaw_iclr2026.tex`: main source; `references.bib`: bibliography.
- `REVISION_SUMMARY_2026-09-03.md`: Korean revision summary and open evidence gaps.
- `reviews/2026-09-03_review_cycles.md`: three internal review rounds with two reviewers.
- `validation_report.json`: final checks; `reviewed_table_provenance.json`: aggregate hashes.

Earlier entries in `REVIEW_LOG.md` and `STRUCTURE_NOTES.md` describe historical drafts
and experiments. The dated revision supersedes their claims about preregistration,
universal superiority, online optimization, judge independence, and completed validation.

## Regenerate reviewed tables

From the repository root:

```bash
python3 scripts/make_paper_tables.py
python3 scripts/make_reviewed_paper_figure.py
```

The legacy generator calls `make_reviewed_paper_tables.py` last, preserving corrected
values and captions. The reviewed generator alone is also runnable. It uses existing
aggregates and invokes no model, training, search, or API. Its provenance file covers
the aggregates it reads, not every raw input behind every historical supplementary table.
Some supplementary numbers remain sourced from existing dedicated analyses.

## Build the PDF

The multilingual appendix now requires XeTeX-compatible font support. Tectonic 0.17.0
was used for this revision; it handles bibliography passes automatically. From the
repository root, with Tectonic on PATH:

```bash
mkdir -p paper/build
tectonic --untrusted --keep-logs --keep-intermediates \
  --outdir paper/build paper/polyjigsaw_iclr2026.tex
```

Alternatively, run XeLaTeX, BibTeX, XeLaTeX, XeLaTeX from `paper/`. Plain pdfLaTeX
is not supported by the new `fontspec` setup. The font files and licenses in `fonts/`
provide CJK, Arabic, and Cyrillic text. Noto Sans CJK came from the official
[notofonts/noto-cjk repository](https://github.com/notofonts/noto-cjk/tree/main/Sans);
DejaVu Sans and its license came from the system font package.

The initial build may download TeX resources. No global TeX installation was changed.
The local revision used `/tmp/polyjigsaw-typesetting/tectonic`; this temporary binary
is not part of the reproducible source package. Read `build/build_console.log` for the
latest build result. Raw-response reconstruction and independent human validation are
not performed by any of these document build steps.
