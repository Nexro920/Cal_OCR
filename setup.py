from setuptools import setup, find_packages

setup(
    name="cal_ocr",
    version="0.1.0",
    author="Nexro",
    description="A CRNN+CTC based OCR model for solving mathematical equation captchas.",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "cal_ocr": ["weights/*.pth"],
    },
    install_requires=[
        "torch>=1.12.0",
        "torchvision",
        "opencv-python",
        "pandas",
        "captcha",
        "Levenshtein",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)