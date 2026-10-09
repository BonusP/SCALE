You are reading one chart panel image from an SOFC paper and writing down the data it
plots, as a table per curve. No pixel coordinates are involved: report values in the
chart's own units, read off its axes.

Return JSON only using the provided schema.

What to identify first:
1. Read the x axis label and unit, the left y axis label and unit, and the right y axis
   label and unit when a right axis exists.
2. Read the axis ranges from the tick labels: the smallest and largest labelled value on
   each axis.
3. List every distinct plotted curve. Use the legend to name them. A legend entry that
   appears twice, once per axis, is two curves — for example one voltage curve and one
   power density curve for the same sample.
4. Decide which axis each curve belongs to. On a current-voltage-power chart the
   monotonically falling curve is voltage on the left axis, and the curve that rises then
   falls is power density on the right axis.

How to read each curve:
- Sample it at 15 to 25 x positions spread across the full range where the curve exists,
  and always include its first and last visible point.
- Add extra samples at anything the shape depends on: a peak, a knee, a steep drop. A
  power density curve must include its maximum.
- Read y by comparing against the gridlines and tick labels around the point, not by
  assuming a straight line between ticks.
- x values must increase along the points list.
- Stop where the curve stops. Do not extrapolate past the last plotted marker, and do
  not invent a point at the axis origin if the curve does not start there.
- Keep the precision the axis supports: about three significant figures is right for a
  0 to 1.2 V axis. Do not report more precision than you can read.

Honesty rules:
- If the curves overlap so much that you cannot follow one of them, still report it, but
  give it a low confidence and say why in its issues.
- If a legend entry has no visible curve, report it with an empty points list and an
  issue.
- If you cannot read an axis range, say so in the top level issues rather than guessing
  a range.
- Never fabricate a curve that is not drawn.

Set curve_role to voltage_curve, power_density_curve, or other. Set axis_binding to
left_y or right_y. Give each curve the legend text as legend_label; if there is no
legend, describe it by colour and marker, such as "red filled circles".
