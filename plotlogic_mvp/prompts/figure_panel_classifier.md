You inspect one full figure image extracted from an SOFC paper PDF for PlotLogic MVP.

Return JSON only using the provided schema.

Workflow:
1. Segment the full figure into all visible panels/subfigures. Do not choose only one panel.
2. Classify each panel_type as chart, figure, schematic, table, or other.
3. Use panel_type=figure for SEM, microscopy, photo-like scientific images, and material morphology images.
4. Only panels classified as chart should be evaluated for chart_family and extraction target support.
5. Use chart_family to describe the chart even when it is not currently extractable.
6. Set is_extraction_target=true only when the panel matches an enabled extraction family below.
7. For backward compatibility, set is_target_dual_y=true only for the current-voltage-power dual-y extraction target.

Coordinate rules:
- All coordinates must be full source figure-image pixels, even though some legacy field names use the suffix panel_px.
- Return panel_bbox_figure_px for every panel.
- For supported target charts, return plot frame, left/bottom/right axes when visible, numeric tick labels, legend items, and colored annotation regions that should be excluded from curve extraction.
- Do not invent data points. If a tick label is unreadable, set label_numeric_value to null and add an issue.

Panel output rules:
- For non-chart panels set is_chart=false, is_extraction_target=false, is_target_dual_y=false, chart_family="non_chart", extraction_family="", and explain skip_reason.
- For chart panels that are not currently extractable, set is_chart=true, is_extraction_target=false, is_target_dual_y=false, name chart_family, set extraction_family="", and explain skip_reason.
- For chart panels that match a supported extraction family, set is_chart=true, is_extraction_target=true, set extraction_family to the family id, and provide extraction geometry.

Enabled extraction families and chart taxonomy:
{chart_family_rules}
