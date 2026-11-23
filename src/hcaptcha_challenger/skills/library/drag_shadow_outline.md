**Task:** Drag the shape(s) from the right panel to their matching outline(s) on the left canvas.

**Grid coordinate rule:**
The second image is a coordinate grid overlay. Its axes are labeled in page-pixel units matching the first image's size.
Always answer in those axis units (page pixels), NOT cell indices.

**Workflow:**
1. Analyze the draggable object(s) on the right panel: silhouette, proportions, key features.
2. Systematically scan every shadow outline on the left canvas; ignore background patterns.
3. High-fidelity matching: find the exact 1:1 match; similar-but-imperfect shapes are decoys.
4. Output one path per draggable object: start_point = draggable object center (typically `<x: 527, y: 189>`), end_point = matching shadow center (read from the grid axes).

**Rules:**
- Shape is signal, background is noise.
- Precision matters: silhouette matching must be exact.
- Every draggable object must have exactly one path in the output.

Output JSON: {"paths": [{"start_point": {"x": <page-pixel int>, "y": <page-pixel int>}, "end_point": {"x": <page-pixel int>, "y": <page-pixel int>}}]}
