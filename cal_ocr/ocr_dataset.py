import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import RandomApply, ColorJitter, GaussianBlur
import cv2

# 保留等号和问号，让 OCR 纯粹做“所见即所得”的识别任务，语义解析交给后处理
VOCAB = "_0123456789+-*/=?"
CHAR2IDX = {char: idx for idx, char in enumerate(VOCAB)}
IDX2CHAR = {idx: char for idx, char in enumerate(VOCAB)}

class CaptchaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, augment=False):
        df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.filenames = df['filename'].tolist()
        self.labels_str = df['label'].tolist()

        base_transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

        if augment:
            aug = RandomApply([
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 0.8)),
            ], p=0.65)   # ← 概率和强度小幅提升
            self.transform = transforms.Compose([aug] + base_transform)
        else:
            self.transform = transform or transforms.Compose(base_transform)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)

        label_str = self.labels_str[idx]
        target = [CHAR2IDX[c] for c in label_str]

        return image, torch.tensor(target, dtype=torch.long), len(target)

def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)

    # 终极修复：使用 1D 拼接代替 2D Padding，彻底避免 CTC Padding 混淆问题
    targets_1d = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, targets_1d, target_lengths

def get_dataloader(csv_file, img_dir, batch_size=128, shuffle=True, num_workers=2, drop_last=True, augment=False):
    dataset = CaptchaDataset(csv_file=csv_file, img_dir=img_dir, augment=augment)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last, pin_memory=True, collate_fn=collate_fn
    )