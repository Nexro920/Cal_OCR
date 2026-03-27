import torch
import torchvision.transforms as transforms
import cv2
import os
import re
import operator

from .ocr_model import CRNN
from .ocr_dataset import VOCAB, IDX2CHAR
from .train import decode_predictions

class CaptchaPredictor:
    def __init__(self, weight_path=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        if weight_path is None:
            # 获取当前 predict.py 所在的目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            weight_path = os.path.join(current_dir, "weights", "best_crnn_model.pth")
        print(f"🚀 初始化 OCR 推理引擎，使用设备: {self.device}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.model = CRNN(num_classes=len(VOCAB)).to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

    def safe_eval(self, math_expr):
        """ 严格的基于正则与栈映射的安全计算器 """
        ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}

        match = re.match(r'^(\d+)([\+\-\*/])(\d+)$', math_expr)
        if not match:
            return f"Error: 语法解析失败 ({math_expr})"

        num1, op, num2 = int(match.group(1)), match.group(2), int(match.group(3))

        if op == '/' and num2 == 0:
            return "Error: 除零错误"

        return int(ops[op](num1, num2))

    def predict(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return "Error", "图片读取失败"

        # 核心修复：强制 Resize，保证高度必然为 60，防止非标图片导致模型崩溃
        image = cv2.resize(image, (160, 60))

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)

        decoded_str = decode_predictions(output)[0]

        math_expr = decoded_str.replace('=', '')
        answer = self.safe_eval(math_expr)

        return decoded_str, answer

if __name__ == '__main__':
    predictor = CaptchaPredictor()
    test_image_path = "math_captchas/val/008001.png"

    if os.path.exists(test_image_path):
        equation, answer = predictor.predict(test_image_path)
        print("-" * 30)
        print(f"🤖 识别公式: {equation}")
        print(f"✅ 最终答案: {answer}")
        print("-" * 30)