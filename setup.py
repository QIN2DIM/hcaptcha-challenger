from pathlib import Path

from setuptools import setup, find_packages

import hcaptcha_challenger

my_readme = Path(__file__).parent.joinpath("README.md").read_text()

# pip install urllib3 -U
# python setup.py sdist bdist_wheel && python -m twine upload dist/*
setup(
    name="hcaptcha-challenger",
    version=hcaptcha_challenger.__version__,
    keywords=["hcaptcha", "hcaptcha-challenger", "hcaptcha-challenger-python", "hcaptcha-solver"],
    author="QIN2DIM",
    author_email="qinse.top@foxmail.com",
    maintainer="QIN2DIM, Bingjie Yan",
    maintainer_email="qinse.top@foxmail.com, bj.yan.pa@qq.com",
    description="🥂 Gracefully face hCaptcha challenge with YOLOv6(ONNX) embedded solution.",
    long_description=my_readme,
    long_description_content_type="text/markdown",
    license="GNU General Public License v3.0",
    url="https://github.com/QIN2DIM/hcaptcha-challenger",
    packages=find_packages(include=["hcaptcha_challenger", "hcaptcha_challenger.*", "LICENSE"]),
    install_requires=[
        "loguru>=0.7.0",
        "opencv-python>=4.8.0.76",
        "numpy>=1.21.5",
        "pyyaml>=6.0",
        "httpx",
    ],
    extras_require={
        "dev": ["nox", "pytest"],
        "test": ["pytest"],
        "selenium": [
            "selenium>=4.11.2",
            "undetected-chromedriver==3.5.2",
            "webdriver-manager==3.8.2",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
)
