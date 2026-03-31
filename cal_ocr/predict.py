import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from .ocr_model import CRNN
from .ocr_dataset import VOCAB, IDX2CHAR

def decode_predictions(preds, blank_idx=0):
    """ CTC 贪心解码，解耦全局变量 """
    _, max_indices = torch.max(preds, dim=2)
    max_indices = max_indices.permute(1, 0).cpu().numpy()

    decoded_strings = []
    for seq in max_indices:
        char_list = []
        for i in range(len(seq)):
            if seq[i] != blank_idx and (not (i > 0 and seq[i - 1] == seq[i])):
                char_list.append(IDX2CHAR[seq[i]])
        decoded_strings.append(''.join(char_list))
    return decoded_strings

class CaptchaPredictor:
    def __init__(self, weights_path=None, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = CRNN(num_classes=len(VOCAB), image_channels=3).to(self.device)

        if weights_path:
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict(self, image_input):
        """ 工业级输入兼容：支持文件路径、NumPy、PIL Image """
        if isinstance(image_input, str):
            image = cv2.imread(image_input, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2 or image_input.shape[2] == 1:
                image = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            else:
                image = image_input
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input.convert("RGB"))
        else:
            raise ValueError("不支持的图片输入格式")

        image = cv2.resize(image, (160, 60), interpolation=cv2.INTER_AREA)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(tensor)
            decoded = decode_predictions(preds)[0]

        return decoded