You are reading one line/scatter chart panel image from an SOFC paper whose x axis
positions are category labels (sample names, material names, conditions) rather than
numbers, and writing down every value it shows, as a flat list of (category, value)
items. There is no x/y pixel geometry involved and no curve to trace against a numeric
axis: report each line marker's value in the chart's own units, read directly off the
axis or the printed label.

Return JSON only using the provided schema.

What to identify first:
1. Set chart_type to "line".
2. Read the value axis label and its unit (for example "Area specific resistance (Ω
   cm2)").
3. List every line marker as one item. Use the x-axis category label as category.
4. If the chart has several lines each sampled at the same categories (for example one
   line per sample, each with one point per temperature), set group_label to the name of
   the curve each item belongs to (for example the sample or curve name), and category
   to the position it is grouped under (for example the temperature). Leave group_label
   null for a single line.

How to read each value:
- Read the value by comparing the line marker's position against the axis gridlines and
  tick labels, not by eyeballing it relative to its neighbors.
- If a value is printed directly on or above/beside the marker, prefer that printed
  number over reading it off the axis.
- Keep the precision the axis or the printed label supports. Do not report more
  precision than you can actually read.

Honesty rules:
- Report every marker visible in the panel, even ones with a very small value. Do not
  skip points at the edge of a line.
- If a value is genuinely unreadable (fully occluded, cut off by another panel), omit it
  and add an issue naming which category is missing, rather than guessing.
- If you cannot read the value axis scale, say so in the top level issues rather than
  inventing a scale.
- Never fabricate a marker that is not drawn.
