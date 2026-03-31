***

# Cal_OCR: 基于 CRNN+CTC 的数学公式验证码识别引擎

Cal_OCR 是一个专为解决**变长、高难度干扰（重度污染）数学验证码**而设计的轻量级光学字符识别（OCR）项目 。该项目采用经典的 `CNN + BiLSTM + CTC Loss` 架构，支持端到端的“所见即所得”字符识别，无需预先切分字符 。

## ✨ 核心特性

* **变长序列识别**：基于 CTC (Connectionist Temporal Classification) 算法，完美支持长度不固定的数学表达式（如 `1+1=?` 或 `99*12=`） 。
* **极致的训练优化**：
    * 引入了 1D 标签拼接技术，彻底解决了 PyTorch 中 CTC Loss Padding 带来的对齐混淆问题 。
    * 使用 `AdamW` 优化器结合 `OneCycleLR` 学习率调度策略，让模型在几十个 Epoch 内迅速收敛 。
    * 原生支持 PyTorch AMP (自动混合精度) 训练，在支持 CUDA 的 GPU 上大幅降低显存占用并提升训练速度 。
* **抗干扰数据管线**：
    * **离线生成**：内置多进程验证码生成脚本，支持随机形变、渐变背景、噪点遮挡、正弦波扭曲等“重度污染”特征 。
    * **在线增强**：DataLoader 中集成了 `ColorJitter` 和 `GaussianBlur`，每次读取图像时动态增加鲁棒性 。
* **工业级推理接口**：`CaptchaPredictor` 类支持直接传入文件路径 (String)、OpenCV 矩阵 (`np.ndarray`) 或 `PIL.Image`，方便无缝接入 Selenium 等自动化爬虫流程 。

---

## 🛠️ 环境依赖与安装

项目依赖 Python 3.x，主要使用 PyTorch 与 OpenCV 。

**1. 克隆或下载本项目**

**2. 安装核心依赖**
为了发挥 GPU 计算的优势，请务必前往 [PyTorch 官网](https://pytorch.org/) 安装与你显卡匹配的 CUDA 版本。
然后安装本项目及其余依赖：
```bash
# 在项目根目录下运行（即包含 setup.py 的目录）
pip install -e .
```
*(这会自动安装 requirements：`torch`, `torchvision`, `opencv-python`, `pandas`, `captcha`, `Levenshtein`, `tqdm`)* 

---

## 🚀 快速上手 (使用指南)

### 第一步：生成抗干扰数据集
模型需要数据进行训练。运行内置的脚本，它将动用多进程在 `math_captchas/` 目录下生成 50,000 张包含各种极端干扰的验证码图片，并按 8:2 自动切分训练集和验证集 。
```bash
python scripts/generate_captchas_img.py
```
*执行完毕后，项目根目录会生成 `math_captchas/` 文件夹，内含 `train/`, `val/` 目录以及对应的 `csv` 标签文件* 。

### 第二步：启动模型训练
确认数据集就绪后，即可开始训练模型。代码会自动检测是否可用 CUDA 加速 。
```bash
python -m cal_ocr.train
```
* **实时监控**：训练过程中会通过 `tqdm` 进度条实时打印 Loss、当前学习率、字符准确率 (Char Acc) 和句子级准确率 (Sent Acc) 。
* **自动保存**：每当验证集准确率突破新高时，模型权重会自动保存至 `cal_ocr/weights/best_crnn_model.pth` 。

### 第三步：模型推理与预测
在爬虫或实际业务中调用训练好的模型非常简单：

```python
import cv2
from cal_ocr import CaptchaPredictor

# 1. 初始化预测器（自动加载 weights/ 目录下的最佳权重，自动分配 GPU/CPU）
# 默认字符集支持 "_0123456789+-*/"
predictor = CaptchaPredictor(weights_path="cal_ocr/weights/best_crnn_model.pth")

# 2. 方式 A：直接传入图片路径
result_str = predictor.predict("test_image.png")
print(f"预测结果: {result_str}")

# 3. 方式 B：传入 OpenCV 读取的内存图像 (适用于 Selenium 截图流)
img_array = cv2.imread("test_image.png")
result_str = predictor.predict(img_array)
print(f"预测结果: {result_str}")
```

---

## 📁 项目结构说明

```text
Cal_OCR/
├── cal_ocr/                 # 核心算法包
│   ├── __init__.py          # 模块导出声明 
│   ├── ocr_dataset.py       # 数据集定义、预处理与在线增强逻辑 
│   ├── ocr_model.py         # CRNN (CNN特征提取 + LSTM序列建模) 网络结构 
│   ├── predict.py           # 推理类与贪心解码逻辑 
│   ├── train.py             # 训练主循环、CER计算与验证逻辑 
│   └── weights/             # 存放训练输出的 .pth 模型权重文件 
├── scripts/
│   └── generate_captchas_img.py  # 验证码生成器 (添加形变、噪点、干扰线) 
├── setup.py                 # 包安装配置 
└── README.md                # 项目文档
```