<div align="center">
    <h1> hCaptcha Challenger</h1>
    <p>🚀 Gracefully face hCaptcha challenge with Yolov5(ONNX) embedded solution.</p>
    <img src="https://img.shields.io/static/v1?message=reference&color=blue&style=for-the-badge&logo=micropython&label=python">
    <img src="https://img.shields.io/github/license/QIN2DIM/hcaptcha-challenger?style=for-the-badge">
    <a href="https://github.com/QIN2DIM/hcaptcha-challenger/releases"><img src="https://img.shields.io/github/downloads/qin2dim/hcaptcha-challenger/total?style=for-the-badge"></a>
	<br>
    <a href="https://github.com/QIN2DIM/hcaptcha-challenger/"><img src="https://img.shields.io/github/stars/QIN2DIM/hcaptcha-challenger?style=social"></a>
	<a href = "https://t.me/joinchat/HlB9SQJubb5VmNU5"><img src="https://img.shields.io/static/v1?style=social&logo=telegram&label=chat&message=studio" ></a>
	<br>
	<br>
</div>

![hcaptcha-challenger-demo](https://github.com/QIN2DIM/img_pool/blob/main/img/hcaptcha-challenger2.gif)

## Introduction

Does not rely on any Tampermonkey script.

Does not use any third-party anti-captcha services.

Just implement some interfaces to make `AI vs AI` possible.

## Requirements

- Python 3.7+
- google-chrome

## Usage

1. **Clone the project code in the way you like.**

2. **Execute the following command in the project root directory.**

   ```bash
   # hcaptcha-challenger
   pip install -r ./requirements.txt
   ```

3. **Download Project Dependencies.**

   The implementation includes downloading the `YOLOv5` object detection model and detecting `google-chrome` in the current environment.

   If `google-chrome` is missing please follow the [prompts](#tour) to download the latest version of the client, if google-chrome is present you need to make sure it is up to date.

   Now you need to execute the `cd` command to access the  `src/` directory of project and execute the following command to download the project dependencies.

   ```bash
   # hcaptcha-challenger/src
   python main.py install
   ```

4. **Start the test program.**

   Check if `chromedriver` is compatible with `google-chrome`.

   ```bash
   # hcaptcha-challenger/src
   python main.py test
   ```

5. **Start the demo program.**

   If the previous test passed perfectly, now is the perfect time to run the demo!

   ```bash
   # hcaptcha-challenger/src
   python main.py demo
   
   # Linux.
   export LC_ALL=zh_CN.UTF8 && export LANG=zh_CN.UTF8 && python3 main.py demo
   ```

## Advanced

1. You can download yolov5 onnx models of different sizes by specifying the `model` parameter in the `install` command.

   - Download `yolov5s6` by default when no parameters are specified. 

   - The models that can be chosen are `yolov5n6`，`yolov5m6`，`yolov5s6`.

   ```bash
   # hcaptcha-challenger/src
   python main.py install --model=yolov5n6
   ```

2. You can run different yolo models by specifying the `model` parameter to compare the performance difference between them.

   - Similarly, when the `model` parameter is not specified, the `yolov5s6` model is used by default.

   - Note that you should use `install` to download the missing models before running the demo.

   ```bash
   # hcaptcha-challenger/src
   python main.py demo --model=yolov5n6
   ```

3. Comparison of programs.

   The following table shows the average solving time of the `hCAPTCHA` challenge for 30 rounds (one round for every 9 challenge images) of mixed categories processed by onnx models of different sizes.

   | model(onnx) | avg_time(s) | size(MB) |
   | :---------: | :---------: | :------: |
   |  yolov5n6   |  **0.71**   | **12.4** |
   |  yolov5s6   |    1.422    |   48.2   |
   |  yolov5m6   |    3.05     |   136    |

   - Use of the `YOLOv5n6(onnx)` embedded scheme to obtain solution speeds close to the limit.

   - Use of the `YOLOv5s6(onnx)` embedded solution, which allows for an optimal balance between stability, power consumption, and solution efficiency.

## Tour

<span id="tour"></span>

### Install Google Chrome on Ubuntu 18.04+

1. Downloading Google Chrome

   ```bash
   wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
   ```

2. Installing Google Chrome

   ```bash
   sudo apt install ./google-chrome-stable_current_amd64.deb
   ```

### Install Google Chrome on CentOS 7/8

1. Start by opening your terminal and downloading the latest Google Chrome `.rpm` package with the following [wget command](https://linuxize.com/post/wget-command-examples/) :

   ```bash
   wget https://dl.google.com/linux/direct/google-chrome-stable_current_x86_64.rpm
   ```

2. Once the file is downloaded, install Google Chrome on your CentOS 7 system by typing:

   ```bash
   sudo yum localinstall google-chrome-stable_current_x86_64.rpm
   ```

### Install Google Chrome on Windows / MacOs

Just go to [Google Chrome](https://www.google.com/chrome/) official website to download and install.

## Reference

- hCaptcha challenge template site [@maximedrn](https://github.com/maximedrn/hcaptcha-solver-python-selenium)
