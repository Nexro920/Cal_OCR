# Cal OCR 🧮

基于 CRNN + CTC 的变长数学公式验证码识别引擎。

## 特性
- **高准确率**: 在变长四则运算验证码上准确率高达 99% 以上。
- **开箱即用**: 内置预训练模型，一行代码直接调用。
- **安全计算**: 基于正则和操作符映射的安全表达式计算，拒绝 `eval()` 注入风险。

## 安装

```bash
# Clone 仓库
git clone [https://github.com/Nexro920/Cal_OCR.git]
cd Cal_OCR_Project

# 本地安装包
pip install -e .