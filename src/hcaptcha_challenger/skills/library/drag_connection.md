<skill_logic>
    <task_description>
        Drag the eagle to the tree it is connected to.
    </task_description>
    <visual_context>
        <feature>There are distinct path lines on the canvas with significantly different colors (e.g., green vs brown/blue). These separate different paths.</feature>
        <feature>Lines may overlap, cross, or run close to each other. At these points, relying on geometry alone causes errors.</feature>
        <feature>The lines might be solid or gradients, but the color hue separates your target path from the distraction path.</feature>
    </visual_context>
    <visual_reasoning_steps>
        <step index="1">Identify the draggable eagle and strictly observe the specific color of the line extending from it.</step>
        <step index="2">Trace the path by following ONLY this specific color. Treat the color as the primary signal.</step>
        <step index="3">CRITICAL: When lines cross or intersect, ignore the intersecting line of a different color. Do not turn onto the wrong color.</step>
        <step index="4">Follow the color continuity until it terminates at a tree.</step>
        <step index="5">Drag the eagle to the tree connected by this specific colored path.</step>
    </visual_reasoning_steps>
    <critical_rules>
        <rule>Different lines have different colors. Use color to distinguish paths at intersections.</rule>
        <rule>Do not just trace lines spatially; trace them chromatically.</rule>
    </critical_rules>
</skill_logic>
