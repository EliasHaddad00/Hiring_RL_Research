# Paper

`main.tex` is the ICML 2026 source (at most 5 pages, excluding references and the contribution statement). Status, Nov 21: problem, approach, protocol and rule results written; introduction and related work are drafts; PPO results, ablations, failure modes and conclusion are marked placeholders (see the header of `main.tex`). Earlier status, Nov 14: skeleton with all section
headings; only Section 3 (problem and simulator) is drafted. It compiles (Nov 21, MiKTeX 25.12, 3 pages; one overfull line). Earlier: there
was no LaTeX installation on the development machine.

## Style files

The ICML 2026 style files are not in the repository. Download
`https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip` and put `icml2026.sty`,
`icml2026.bst`, `fancyhdr.sty`, `algorithm.sty` and `algorithmic.sty` next to `main.tex`
(or upload the zip's contents with `main.tex` and `references.bib` to Overleaf).

## Build

```
latexmk -pdf main.tex
```

## Rules

- A reference goes into `references.bib` only after the paper has been opened and read.
- Figures are vector files written by `experiments/analysis/` into `paper/figures/`, tables into `paper/tables/`.
  `python experiments/analysis/make_figures.py` regenerates all of them from saved results (`make figures`
  calls the same script). Cells show `--` until the final runs are scored; `tables/prov.tex` removes the
  provisional mark once the rules have evaluation-pool scores.
- Every number in the text comes from a file under `results/`; prototype numbers are never results.
