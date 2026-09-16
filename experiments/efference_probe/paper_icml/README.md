# ICML draft: Novelty-Gated Compensation

LaTeX source, built PDF, and figure pipeline for the conference draft. All
numbers come from the two collection waves of this experiment directory:
wave 1 (`main01` + the 50-episode mirror/freeze arms) and wave 2
(`main02`, expanded mirror/freeze, `suite_spatial02`, `suite_object01`,
`suite_long01`; see `../reports/p0_wave2_report.md` and `../logbook.md`).

## Build

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Uses the ICML 2026 style kit (`icml2026.sty`/`.bst`, included); swap in the
official ICML 2027 kit when released.

## Figures

Static vector PDFs are checked in. To regenerate:

- `make_figs_icml_w2.py` rebuilds `fig_headline.pdf`, `fig_ladder.pdf`,
  `fig_e3_logprob.pdf` from the run exports. Point it at the data with
  `EFP_WAVE1_DATA=<dir with main01/> EFP_WAVE2_DATA=<dir with main02/ ...>`;
  each run directory needs `calls.parquet` and `analysis*/` (the export
  tarballs, not `hidden/`). Every E3 mean is recomputed from the raw
  exports and asserted against the pipeline values before anything is
  drawn.
- `make_figs_icml.py` is the wave-1 script that produced the unchanged
  figures (`fig_setup.pdf`, `fig_e1.pdf`, `fig_frames.pdf`); kept for
  provenance.

Neither script is part of the experiment pipeline; probing outputs are
produced by `../run_probes.py`.
