# Figure QA

- Core conclusion: same-target S2 utility is directionally associated with S3
  utility, but the evidence is too unstable to authorize SCAA development.
- Evidence chain: the transfer scatter shows effect direction and uncertainty;
  the policy panel shows negligible mean policy separation; harm/coverage shows
  that coverage is nontrivial but harm reduction is weak; the per-subject panel
  exposes sign reversals rather than hiding them in an average.
- Archetype: quantitative validation grid delivered as four standalone figures.
- Backend: Python/matplotlib exclusively.
- Data integrity: every one of the 41 development subjects is shown; seeds are
  averaged within subject and backbone before plotting; no subject is excluded.
- Source data: compact CSVs in `results/`; inference is defined in the protocol
  lock and final report.
- Export: 183-mm-wide editable SVG/PDF plus 600-dpi PNG. PNG is retained instead
  of TIFF to keep the repository compact; the PDF/SVG are the publication
  masters.
- Visual checks: white background, consistent sans-serif type, non-rainbow
  palette, shape/hatch redundancy, readable labels, no clipping or overlaps.
