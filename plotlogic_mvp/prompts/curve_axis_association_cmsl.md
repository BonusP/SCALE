<!-- ###수정 차이 18: 추출 완료된 curve ID와 left/right y-axis 사이의 association을 판단합니다. -->
Inspect the supplied chart overlay and metadata. Return JSON using the supplied schema.

The overlay marks each EXTRACTED curve as T1, T2, etc.; the color key maps those markers to trace_id. Identify which y-axis each extracted curve uses from the original chart's axis titles, legend labels, curve color, arrows and visible scientific meaning. Overlay colors are identifiers, not necessarily original curve colors.

Return exactly one assignment per supplied trace_id, using axis_binding=left_y or right_y. A single legend condition can name both a voltage and a power-density curve; decide for each trace separately. Do not assume a curve maps to the left axis merely because an earlier legend hint was missing. If you cannot determine the association, set axis_binding=null, explain why and add an issue. Confidence must lie in [0,1]. Do not invent, remove, smooth or refit data points.

<!-- ###수정 추가 수정: shape 필터 전에 모든 후보를 판정하고 같은 색의 V/P label을 각각 연결합니다. -->
These are GEOMETRIC curve candidates before any semantic role filtering. A power curve can be continuously increasing without a visible peak. Open-circle markers can represent power, while connected filled markers of the SAME original color can represent voltage. Follow each overlay's actual path and original marker style, not an earlier role guess.

For each assignment also return legend_label and is_curve. Choose legend_label ONLY from that trace's legend_options (or its single supplied legend_label). If two options share the same RGB, use the actual curve's axis, line style, marker style and legend meaning to select V versus P; do not assign the first option to every curve. If no unique label is supported, return null.

Set is_curve=false for an arrow, axis/grid artifact, or a track that incorrectly jumps between separate voltage and power curves at an intersection. Set axis_binding=null in such cases. A short but coherent segment of a real curve can still be valid; explain limited coverage. Overlay identifier colors do not equal original series colors. Supplied tracking support may interpolate bounded gaps between aligned strokes; it is not evidence for switching branches. Do not fill missing data yourself.
<!-- ### -->
