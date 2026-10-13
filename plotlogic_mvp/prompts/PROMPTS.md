# Prompts

| File | Used by | Purpose |
| --- | --- | --- |
| `figure_panel_classifier_cmsl.md` | `app/services/vision.py` | Splits a figure into panels and returns chart family, axes, ticks, and legend geometry |
| `chart_families.json` | `app/services/vision.py` | Chart-family registry rendered into the classifier prompt (`{chart_family_rules}`) |
| `curve_axis_association_cmsl.md` | `app/services/curve_axis_association.py` | Assigns each traced curve to the left or right y-axis on dual-y charts |
| `categorical_chart_direct.md` | `app/services/codex_direct.py` | Fallback that reads category/value pairs when pixel tracing of a categorical chart is incomplete |
| `visual_qa/mask_quality.md`, `visual_qa/trace_overlay.md` | `app/services/visual_qa.py` | Reference prompts recorded with the visual QA artifacts |

## Prompt hash

Every classification records `prompt_sha256` in `prompt_metadata` inside `candidates/*.json`. The hash covers the rendered classifier prompt plus `chart_families.json`, so editing either one changes it. A re-run reuses a previous classification only when the hash matches, so editing the prompt or the registry forces reclassification.

| Template | Registry version | prompt_sha256 |
| --- | --- | --- |
| `figure_panel_classifier_cmsl.md` | `2026-10-07.four-line-families.categorical-tracing` | `5d7fe1c9201293a10f7859636e91f9a9b9619af590fc891ee6edb3d5feba4080` |

## Chart families

`chart_families.json` enables four line-chart families: `current_voltage_power_dual_y`, `single_axis_line_chart`, `dual_y_generic_line_chart`, and `categorical_line_chart`. Visible connecting lines are required; markers on lines are allowed, while marker-only scatter plots are excluded. Bar and pie charts are listed as non-target families.
