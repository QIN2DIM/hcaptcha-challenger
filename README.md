<div align="center">
    <h1> hCaptcha Challenger</h1>
    <p>🚀 Gracefully face hCaptcha challenge with MoE(ONNX) embedded solution.</p>
    <img src="https://img.shields.io/static/v1?message=reference&color=blue&style=for-the-badge&logo=micropython&label=python">
    <img src="https://img.shields.io/github/license/QIN2DIM/hcaptcha-challenger?style=for-the-badge">
    <a href="https://github.com/QIN2DIM/hcaptcha-challenger/releases"><img src="https://img.shields.io/github/downloads/qin2dim/hcaptcha-challenger/total?style=for-the-badge"></a>
	<br>
	<a href="https://discord.gg/KA66wZBQ"><img alt="Discord" src="https://img.shields.io/discord/978108215499816980?style=social&logo=discord&label=echosec"></a>
 	<a href = "https://t.me/+Cn-KBOTCaWNmNGNh"><img src="https://img.shields.io/static/v1?style=social&logo=telegram&label=chat&message=studio" ></a>
	<br>
	<br>
</div>


![hcaptcha-challenger-demo](https://github.com/QIN2DIM/img_pool/blob/main/img/hcaptcha-challenger3.gif)

## Introduction

Does not rely on any Tampermonkey script.

Does not use any third-party anti-captcha services.

Just implement some interfaces to make `AI vs AI` possible.

## Requirements

- Python 3.8 or higher.
- Windows 10+, Windows Server 2016+ or Windows Subsystem for Linux (WSL).
- MacOS 12 Monterey or MacOS 13 Ventura.
- Debian 11, Debian 12, Ubuntu 20.04 or Ubuntu 22.04.

## What's features

| Challenge Type                          | Pluggable Resource                                                     |
| --------------------------------------- | ------------------------------------------------------------ |
| `image_label_binary`                    | ResNet ONNX [#challenge](https://github.com/QIN2DIM/hcaptcha-challenger/issues?q=label%3A%22%F0%9F%94%A5+challenge%22+) |
| `image_label_area_select: point`        | YOLOv8s ONNX [#588](https://github.com/QIN2DIM/hcaptcha-challenger/issues/588)                                           |
| `image_label_area_select: bounding box` | YOLOv8m ONNX [#592](https://github.com/QIN2DIM/hcaptcha-challenger/issues/592)                                           |

## Reference

- [beiyuouo/hcaptcha-model-factory 🏗](https://github.com/beiyuouo/hcaptcha-model-factory)
- [roboflow motion](https://universe.roboflow.com/qin2dim/hcaptcha-challenger)
- [captcha-challenge/datasets 🦜](https://github.com/captcha-challenger/hcaptcha-whistleblower)
- [ultralytics/YOLOv8 🚀 ](https://github.com/ultralytics/ultralytics)
- [ultrafunkamsterdam/undetected-chromedriver](https://github.com/ultrafunkamsterdam/undetected-chromedriver)
- [microsoft/playwright-python](https://github.com/microsoft/playwright-python)
- hCaptcha challenge template site [@maximedrn](https://github.com/maximedrn/hcaptcha-solver-python-selenium)
