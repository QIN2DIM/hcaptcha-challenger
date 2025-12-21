# Instructions

Your task is to classify challenge questions into one of four types:

1. `image_label_single_select`: Requires clicking on a SINGLE specific area/object of an image based on a prompt
2. `image_label_multi_select`: Requires clicking on MULTIPLE areas/objects of an image based on a prompt
3. `image_drag_single`: Requires dragging a SINGLE puzzle piece/element to a specific location on an image
4. `image_drag_multi`: Requires dragging MULTIPLE puzzle pieces/elements to specific locations on an image

## Rules

- Output ONLY one of the four classification types listed above
- Do not provide any explanations, reasoning, or additional text
- For clicking/selecting tasks:
  - If the question implies selecting ONE item/area, output `image_label_single_select`
  - If the question implies selecting MULTIPLE items/areas, output `image_label_multi_select`
  - IF the question implies 9grid selection, output `image_label_multi_select`
- For dragging tasks:
  - If the question implies dragging ONE item/element, output `image_drag_single`
  - If the question implies dragging MULTIPLE items/elements, output `image_drag_multi`

## Examples

Input: "Please click on the object that is different from the others"
Output: `image_label_single_select`

Input: "Please click on the two elements that are identical"
Output: `image_label_multi_select`

Input: "Please drag the puzzle piece to complete the image"
Output: `image_drag_single`

Input: "Arrange all the shapes by dragging them to their matching outlines"
Output: `image_drag_multi`
