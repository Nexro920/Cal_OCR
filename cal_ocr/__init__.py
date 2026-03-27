# cal_ocr/__init__.py
from .predict import CaptchaPredictor
from .ocr_model import CRNN

__version__ = "0.1.0"
__all__ = ["CaptchaPredictor", "CRNN"]