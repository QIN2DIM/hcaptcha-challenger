from setuptools import setup, find_packages
from os import path as os_path
import hcaptcha_challenger

this_directory = os_path.abspath(os_path.dirname(__file__))

setup(name="hcaptcha-challenger",
      version=hcaptcha_challenger.__version__,
      keywords=[
          "hcaptcha",
          "hcaptcha-challenger",
          "hcaptcha-challenger-python",
      ],
      author="QIN2DIM",
      author_email="qinse.top@foxmail.com",
      maintainer="QIN2DIM, Bingjie Yan",
      maintainer_email="qinse.top@foxmail.com, bj.yan.pa@qq.com",
      description="🥂 Gracefully face hCaptcha challenge with Yolov5(ONNX) embedded solution.",
      long_description=open(os_path.join(this_directory, "README.md")).read(),
      long_description_content_type="text/markdown",
      license="GNU GENERAL PUBLIC LICENSE",
      url="https://github.com/QIN2DIM/hcaptcha-challenger",
      packages=find_packages(include=['hcaptcha_challenger', 'hcaptcha_challenger.*', 'LICENSE']),
      install_requires=[
          "fire~=0.4.0", "loguru~=0.6.0", "selenium~=4.1.0", "aiohttp~=3.8.1",
          "opencv-python~=4.5.5.62", "undetected_chromedriver==3.1.3", "webdriver-manager>=3.5.2",
          "scikit-image~=0.19.2", "numpy>=1.21.5", "requests~=2.27.1", "pyyaml~=6.0"
      ],
      extras_require={
          "dev": ["nox", "pytest"],
          "test": ["pytest"],
      },
      python_requires='>=3.7',
      classifiers=[
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development', 'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3'
      ])
