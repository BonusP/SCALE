<!-- chart_fmailiy 다시 4개로 수정: shared four-family registry and current prompt hashes. -->
# Vision prompt templates

`figure_panel_classifier.md` is the **baseline** and must never be edited: it is the
comparison point for every prompt experiment. A variant goes in a new file, so the
baseline keeps producing baseline results.

Pick the prompt per run — the **Vision prompt** dropdown above the paper list, or the
`prompt_variant` query parameter:

```bash
curl -X POST "localhost:8000/api/paperrag/papers/9/extract?prompt_variant=figure_panel_classifier_bbox_v2.md"
curl "localhost:8000/api/prompts"          # available templates and their hashes
```

`PLOTLOGIC_MVP_VISION_PROMPT` changes the default that the dropdown starts on; an
explicit per-run choice always wins. An unknown or path-like name is rejected with 400
rather than silently falling back to another prompt.

Every run records its choice in two places: `jobs.prompt_variant` (shown in the job
header) and `prompt_metadata` inside each `candidates/*.json`, alongside
`prompt_sha256`. The hash covers the *rendered* prompt plus `chart_families.json`, so
editing the family registry changes it too.

| Template | prompt_sha256 | Notes |
| --- | --- | --- |
| `figure_panel_classifier.md` | `e9a7a57b9f667503c468f130231ed2445884c19038b307f4608d5465ac511284` | Baseline. Classification and coordinate rules only; no guidance on bbox precision. **Do not edit the template body.** Hash has moved four times: `registry_version 2026-09-16.mvp2` added `single_axis_line_chart` and `dual_y_generic_line_chart` (plus an `axis_binding_hint` instruction on `current_voltage_power_dual_y`); `2026-09-16.mvp3` added `bar_chart` and `pie_chart`; `2026-09-16.mvp4` added `categorical_line_chart`; `2026-09-17.mvp5` removed `bar_chart` and `pie_chart` again (extraction is colored line charts only — single-axis or dual-axis — with a categorical-x line chart still routed through the categorical direct-read path) and tightened the line families to require curves distinguishable by color. Each bump intentionally invalidates the candidate cache so already-classified figures get re-classified against the current family list. |
| `figure_panel_classifier_bbox_v2.md` | `d74d3a98d5c68d6c3ea483351fcc0fec249407fbab1a823318b0ab5cadd15fb9` | Variant, used by the `ocr_v2` method. Adds per-box placement rules, an image-bounds check, tick-mark-based `tick_position_px`, a pre-return self-check, and (2026-09-16) a legend `axis_binding_hint` instruction. States that text boxes are refined by a later OCR pass so only coarse accuracy is needed there. |

## Other prompts

`curve_table_direct.md` is not a panel classifier and is not selected by the variable
above. The `codex_direct` method uses it after the panel crop to ask codex for each
curve's values in the chart's own units, replacing the pixel extraction entirely.

<!-- categorical 처리를 위한 수정 -->
`categorical_chart_direct.md` is a table-reading fallback rather than a panel classifier.
For `cmsl_minimal`, categorical charts first generate color filters and pixel traces.
Category tick positions and y calibration convert supported traces into (category, value)
pairs. Missing geometry, missing series or ambiguous samples trigger Codex reading with
both the original panel and a trace overlay. Other methods retain direct reading.
Categorical CSV columns remain `panel_id, series_id, legend_label, category, value`.

Bar charts and pie charts are not extraction targets: `chart_families.json` only lists
colored line/scatter chart families (single-axis or dual-axis) as
`enabled_extraction_families`, and `bar_chart`/`pie_chart` are listed in
`known_non_target_chart_families` instead.

## Adding a variant

1. Copy the baseline to `figure_panel_classifier_<name>.md` and edit the copy.
2. Add a row above with its sha (`prompt_sha256_for` in `app/services/vision.py`).
3. It appears in the dropdown automatically; no restart needed.

`test_baseline_prompt_is_unchanged` pins the baseline hash and fails if the baseline is
touched. `test_prompt_variants_are_documented` fails if a template is missing from the
table above. `test_prompt_variant_is_recorded_per_run` fails if a run stops recording
which prompt it used.

Switching prompts invalidates the candidate cache: `_find_cached_figure_analysis` only
reuses a previous job's classification when the recorded sha matches, so a variant run
re-calls codex for every figure image instead of reusing baseline results.

<!-- ###수정 차이 03·04·18: 실험용 prompt와 registry별 hash를 문서화합니다. -->

- figure_panel_classifier.md: e9a7a57b9f667503c468f130231ed2445884c19038b307f4608d5465ac511284
- figure_panel_classifier_bbox_v2.md: d74d3a98d5c68d6c3ea483351fcc0fec249407fbab1a823318b0ab5cadd15fb9
- figure_panel_classifier_cmsl.md: 5d7fe1c9201293a10f7859636e91f9a9b9619af590fc891ee6edb3d5feba4080

<!-- chart_fmailiy 다시 4개로 수정 -->
CMSL now uses `chart_families.json` with four families: `current_voltage_power_dual_y`,
`single_axis_line_chart`, `dual_y_generic_line_chart`, and `categorical_line_chart`.
Visible connecting lines are required; markers on lines are allowed, marker-only scatter is excluded.
Single-axis and categorical charts may have one or more series. Numeric-axis charts retain
the CMSL pixel-tracing pipeline; CMSL categorical charts also trace pixels, with Codex table fallback.
`chart_families_cmsl.json` is retained as an inactive reference.
The shared registry change updates baseline and bbox-v2 hashes too; their template bodies are unchanged.
`curve_axis_association_cmsl.md` remains the dual-y post-tracking axis assignment prompt.

<!-- ###수정 추가 수정: 최종 role 선택 전 curve별 축·범례·curve 여부를 판정하는 prompt hash를 기록합니다. -->
- curve_axis_association_cmsl.md: f9f8cffa12960d9089676c83688273dd28cb8664d365324d0fbd453f51f34eaf

The axis prompt receives geometric candidates before semantic role filtering and returns axis_binding, legend_label and is_curve for each candidate.
<!-- ### -->

<!-- categorical 처리를 위한 수정 -->
CMSL categorical charts now produce color filters, masks and pixel traces before table conversion.
Category tick positions and numeric y calibration are requested from Vision. Values with local trace
support are converted in code; missing series, ambiguous values or missing geometry trigger Codex
reading with the original panel and trace overlay. Debug charts display category labels on x.
Baseline/ocr-v2 categorical runs retain direct reading. Earlier direct-read descriptions describe
the previous CMSL behavior. Saved runs are not regenerated automatically.
