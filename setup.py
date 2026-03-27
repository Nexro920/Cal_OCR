from setuptools import setup, find_packages

setup(
    name="cal_ocr", # 用户 pip install 的名字
    version="0.1.0",
    author="Your Name",
    description="A CRNN+CTC based OCR model for solving mathematical equation captchas.",
    packages=find_packages(), # 自动寻找 cal_ocr 文件夹
    include_package_data=True, # 允许打包非 .py 文件
    package_data={
        # 这一行确保打包时将权重文件带上
        "cal_ocr": ["weights/*.pth"],
    },
    install_requires=[
        "torch>=1.9.0",
        "torchvision",
        "opencv-python",
        "pandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)