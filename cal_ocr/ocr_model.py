import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, image_channels=1):
        super(CRNN, self).__init__()

        # CNN 骨干网络：高度缩减至 1，宽度保留为时间步
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 60x160 -> 30x80

            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 30x80 -> 15x40

            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # 15x40 -> 7x40 (高度减半，宽度保持)

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((7, 1), (7, 1)) # 7x40 -> 1x40 (高度压扁为1，时间步T=40)
        )

        # RNN 序列网络
        self.rnn = nn.LSTM(input_size=256, hidden_size=128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        # 1. CNN 提取特征
        conv = self.cnn(x) # [Batch, 256, 1, 40]

        # 2. 形状变换适配 RNN：[Batch, Channels, Height, Width] -> [Batch, Width, Channels]
        b, c, h, w = conv.size()
        conv = conv.squeeze(2) # [Batch, 256, 40]
        conv = conv.permute(0, 2, 1) # [Batch, 40, 256] (即 [Batch, T, Features])

        # 3. RNN 处理序列
        rnn_out, _ = self.rnn(conv) # [Batch, 40, 256]

        # 4. 全连接映射到类别
        output = self.fc(rnn_out) # [Batch, 40, num_classes]

        # PyTorch 的 CTCLoss 默认期望 [T, Batch, Num_classes]
        output = output.permute(1, 0, 2)

        # CTCLoss 要求输入是对数概率
        return nn.functional.log_softmax(output, dim=2)