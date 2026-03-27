import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import cv2

# CTC 必须要有一个占位用的空白符 (Blank)，放在 index 0。将原本的 '-' 改为 '_' 避免与减号冲突。
VOCAB = "_0123456789+-*/="
CHAR2IDX = {char: idx for idx, char in enumerate(VOCAB)}
IDX2CHAR = {idx: char for idx, char in enumerate(VOCAB)}

class CaptchaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.filenames = df['filename'].tolist()
        self.labels_str = df['label'].tolist()

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])

        # 使用 OpenCV 读取单通道灰度图
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image)

        label_str = self.labels_str[idx]
        target = [CHAR2IDX[c] for c in label_str]
        target_length = len(target)

        # 直接返回真实长度的 target tensor，Padding 交给 collate_fn 处理
        return image, torch.tensor(target, dtype=torch.long), target_length

def collate_fn(batch):
    """动态 Padding，将同一 Batch 内的标签补齐到该 Batch 的最大长度"""
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    # 使用空白符 (0) 进行填充
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return images, targets, target_lengths

def get_dataloader(csv_file, img_dir, batch_size=64, shuffle=True, num_workers=4, drop_last=True):
    dataset = CaptchaDataset(csv_file=csv_file, img_dir=img_dir)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last, pin_memory=True, collate_fn=collate_fn
    )