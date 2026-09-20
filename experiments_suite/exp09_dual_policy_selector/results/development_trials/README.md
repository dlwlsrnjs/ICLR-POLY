# Development-only replay artifacts

These files preserve intermediate experiments: pre-signature, response-transition, GP, OOD, and
long-budget smoke runs. They are not the frozen result because the online simulator subsequently
fixed duplicate harmless-probe reuse, invalid-as-failure handling, and full-evidence conditioning.

Use these only as an implementation audit trail. The current frozen selection is in:

- `../separate_axis_pipeline_search.json`
- `../selected_pipeline_global.json`
- `../selected_pipeline_mj.json`
- `../selected_pipeline_lg.json`
- `../OPTIMAL_PIPELINE_REPORT.md`
