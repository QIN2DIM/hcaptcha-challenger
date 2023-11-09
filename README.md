<div align="center">
    <h1> hCaptcha Challenger</h1>
    <p>🚀 Gracefully face hCaptcha challenge with MoE(ONNX) embedded solution.</p>
    <img src="https://img.shields.io/pypi/v/hcaptcha-challenger?style=flat-square&logo=python&logoColor=white">
    <img src="https://img.shields.io/pypi/dw/hcaptcha-challenger?style=flat-square&logo=aiqfome&label=downloads%40PyPI">
    <a href="https://github.com/QIN2DIM/hcaptcha-challenger/releases"><img src="https://img.shields.io/github/downloads/QIN2DIM/hcaptcha-challenger/model/total?style=flat-square&logo=github"></a>
	<br>
	<a href="https://discord.gg/m9ZRBTZvbr"><img alt="Discord" src="https://img.shields.io/discord/978108215499816980?style=social&logo=discord&label=echosec"></a>
 	<a href = "https://t.me/+Cn-KBOTCaWNmNGNh"><img src="https://img.shields.io/static/v1?style=social&logo=telegram&label=chat&message=studio" ></a>
	<br>
	<br>
</div>


![hcaptcha-challenger-demo](https://github.com/QIN2DIM/img_pool/blob/main/img/hcaptcha-challenger3.gif)

## Introduction

Does not rely on any Tampermonkey script.

Does not use any third-party anti-captcha services.

Just implement some interfaces to make `AI vs AI` possible.

## What's features

| Challenge Type                          | Pluggable Resource                                           |
| --------------------------------------- | ------------------------------------------------------------ |
| `image_label_binary`                    | ResNet  ONNX classification [#challenge](https://github.com/QIN2DIM/hcaptcha-challenger/issues?q=label%3A%22%F0%9F%94%A5+challenge%22+) |
| `image_label_area_select: point`        | YOLOv8 ONNX detection  [#588](https://github.com/QIN2DIM/hcaptcha-challenger/issues/588) |
| `image_label_area_select: bounding box` | YOLOv8 ONNX segmentation  [#592](https://github.com/QIN2DIM/hcaptcha-challenger/issues/592) |
| `image_label_multiple_choice`           | ViT ONNX zero-shot motion [#917](https://github.com/QIN2DIM/hcaptcha-challenger/issues/917) |

| Advanced Task               | Pluggable Resource                                           |
| --------------------------- | ------------------------------------------------------------ |
| `Rank.Strategy`             | [#nested-model-zoo](https://github.com/QIN2DIM/hcaptcha-challenger/issues/797) |
| `self-supervised challenge` | [#CLIP-ViT](https://github.com/QIN2DIM/hcaptcha-challenger/issues/858) |

## Workflow

| Tasks                         | Resource                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| `ci: sentinel`                | [![hCAPTCHA Sentinel](https://github.com/QIN2DIM/hcaptcha-challenger/actions/workflows/sentinel.yaml/badge.svg?branch=main)](https://github.com/QIN2DIM/hcaptcha-challenger/actions/workflows/sentinel.yaml) |
| `ci: collector`               | [![hCAPTCHA Collector](https://github.com/QIN2DIM/hcaptcha-challenger/actions/workflows/collector.yaml/badge.svg)](https://github.com/QIN2DIM/hcaptcha-challenger/actions/workflows/collector.yaml) |
| `datasets: VCS, annoate`      | [#roboflow](https://app.roboflow.com/), [#model-factory](https://github.com/beiyuouo/hcaptcha-model-factory) |
| `model: ResNet - train / val` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/captcha-challenger/hcaptcha-model-factory/blob/main/automation/roboflow_resnet.ipynb) |
| `model: YOLOv8 - train / val` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QIN2DIM/hcaptcha-challenger/blob/main/automation/roboflow_yolov8.ipynb) |
| `model: upload, upgrade`      | [#objects](https://github.com/QIN2DIM/hcaptcha-challenger/tree/main/src), [#modelhub](https://github.com/QIN2DIM/hcaptcha-challenger/releases/tag/model) |
| `datasets: public, archive`   | [#roboflow-universe](https://universe.roboflow.com/qin2dim/), [#captcha-datasets](https://github.com/captcha-challenger/hcaptcha-whistleblower) |

## Contributors
I would like to express my sincere gratitude to all the contributors.

[![](https://opencollective.com/hcaptcha-challenger/contributors.svg?width=890&button=false)](https://github.com/QIN2DIM/hcaptcha-challenger/graphs/contributors)

## What's next

- [Dislock](https://github.com/Vinyzu/DiscordGenerator), the most advanced Discord Browser Generator. Powered by hCaptcha Solving AI.
- [undetected-playwright](https://github.com/QIN2DIM/undetected-playwright), stash the fingerprint of playwright-based web agents.
- [epic-awesome-gamer](https://github.com/QIN2DIM/epic-awesome-gamer), gracefully claim weekly free games from Epic Store.

## Reference

- [microsoft/playwright-python](https://github.com/microsoft/playwright-python)
- [ultrafunkamsterdam/undetected-chromedriver](https://github.com/ultrafunkamsterdam/undetected-chromedriver)
- hCaptcha challenge template site [@maximedrn](https://github.com/maximedrn/hcaptcha-solver-python-selenium)
