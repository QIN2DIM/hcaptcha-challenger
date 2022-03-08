---
name: i18n challenge
about: i18n challenge
title: 'feat(pending): `i18n` challenge | [langName]'
labels: i18n
assignees: ''

---

<!--Modify [langName] in the title-->
<!--i18n challenge | ru-->
<!--i18n challenge | jp-->

## Last modified time

<!--The UTC time of your first issue submission-->

## Prompt message

<!--Prompt message, which must contain all known labels.  It is critical that the prompt information be copied from the web page tabs rather than entered manually.-->

```markdown
<!--truck-->
Please click each image containing a truck
<!--boat-->
Please click each image containing a boat
<!--bicycle-->
Please click each image containing a bicycle
<!--train-->
Please click each image containing a train
<!--seaplane-->
Please click each image containing a seaplane
<!--aeroplane-->
Please click each image containing an airplane
<!--bus-->
Please click each image containing a motorbus
<!--motorbike-->
Please click each image containing a motorcycle
<!--vertical river-->
Please click each image containing a vertical river
<!--airplane in the sky flying left-->
Please click each image containing an airplane in the sky flying left
```

## Screenshot of prompt message[optional]

<!--Contains prompt messages and the challenge sample-->

## Label alias

<!--Modify the key in the dictionary-->

```python
label_alias = {
    "truck": "truck",
    "boat": "boat",
    "bicycle": "bicycle",
    "train": "train",
    "airplane": "aeroplane",
    "motorbus": "bus",
    "motorcycle": "motorbike",
    "vertical river": "vertical river",
    "airplane in the sky flying left": "airplane in the sky flying left",
}

```

## Split function

<!--Must be elegant and refined enough to work with all labels-->

```python
labels_mirror = {
    "zh": re.split(r"[包含 图片]", label_obj.text)[2][:-1],
    "en": re.split(r"containing a", label_obj.text)[-1][1:].strip(),
    "ru": ...
    "jp": ...
}
label = labels_mirror[langName]
```
