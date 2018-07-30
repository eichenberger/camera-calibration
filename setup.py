import os
from setuptools import setup
setup(
    name = "camera-calibration",
    version = "1.0.0",
    author = "Stefan Eichenberger",
    author_email = "eichest@gmail.com",
    description = ("An example on how to calibrate a camera based on a 3D cloud"),
    license = "MIT",
    keywords = "camera-calibration, computer vision",
    install_requires=[
        'numpy >= 1.14.1',
        'scipy >= 0.19.1',
        'opencv-python >= 3.2.0'
    ]
)
