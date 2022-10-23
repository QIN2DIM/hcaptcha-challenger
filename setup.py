import os

from setuptools import setup, find_packages
import hcaptcha_challenger

this_directory = os.path.abspath(os.path.dirname(__file__))

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
    long_description=open(os.path.join(this_directory, "README.md")).read(),
    long_description_content_type="text/markdown",
    license="GNU General Public License v3.0",
    url="https://github.com/QIN2DIM/hcaptcha-challenger",
    packages=find_packages(include=["hcaptcha_challenger", "hcaptcha_challenger.*", "LICENSE"]),
    install_requires=[
        "loguru~=0.6.0",
        "selenium~=4.4.3",
        "aiohttp~=3.8.1",
        "opencv-python~=4.5.5.62",
        "undetected-chromedriver==3.1.5.post4",
        "webdriver-manager==3.8.2",
        "numpy>=1.21.5",
        "requests>=2.28.1",
        "pyyaml~=6.0",
    ],
    extras_require={"dev": ["nox", "pytest"], "test": ["pytest"]},
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
