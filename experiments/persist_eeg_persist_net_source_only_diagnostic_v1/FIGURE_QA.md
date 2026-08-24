# Figure QA

- Core conclusion: the frozen PUD source representation is below Vanilla EEGNet and the capacity-matched dual control; target adaptation provides only a small partial recovery.
- Archetype: quantitative comparison figures with one claim-bearing panel per export.
- Backend: Python/matplotlib only for drawing, rendering, and export.
- Source data: `results/source_only_main.csv`, `results/per_subject_results.csv`, and `results/source_vs_adapted.csv`.
- Data inclusion: all six source-only methods and all 40 outcome subjects are shown where applicable; no observation was sampled or excluded.
- Protocol: OpenBMI V8_SEARCH, 40 subjects, five folds, three seeds, future Session 2.
- Metric: mean subject balanced accuracy; seed values are averaged within subject before subject-level summaries.
- Intervals: 10,000-draw paired subject bootstrap where shown.
- Export: PNG at 300 dpi plus PDF/SVG with editable text; white background and sans-serif fallback.
- Final dimensions: 7.2 × 4.2 inches (approximately 183 × 107 mm) before tight bounding-box adjustment.
- Automated source preflight: no FAIL findings after adding SVG export. The TIFF warning is accepted because the requested deliverables are PNG/PDF and both PDF/SVG provide vector submission masters; the random-number warning refers only to preregistered bootstrap/sign-flip resampling, not simulated plot data.
- Visual inspection: labels, error bars, zero reference, and color/direction encoding must be checked on the regenerated PNG/PDF files after finalization.
