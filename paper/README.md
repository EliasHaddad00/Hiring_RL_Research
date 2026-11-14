# Paper

`main.tex` is the ICML 2026 source (at most 5 pages). Status, Nov 14: skeleton with all section
headings; only Section 3 (problem and simulator) is drafted. It has not been compiled yet: there
is no LaTeX installation on the development machine.

## Style files

The ICML 2026 style files are not in the repository. Download
`https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip` and put `icml2026.sty`,
`icml2026.bst`, `fancyhdr.sty`, `algorithm.sty` and `algorithmic.sty` next to `main.tex`
(or upload the zip's contents with `main.tex` and `references.bib` to Overleaf).

## Build

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Rules

- A reference goes into `references.bib` only after the paper has been opened and read.
- Figures are vector files written by `experiments/analysis/` into `paper/figures/`.
- Every number in the text comes from a file under `results/`; prototype numbers are never results.
