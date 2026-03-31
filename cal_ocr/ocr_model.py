import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, image_channels=3):
        super().__init__()

        # 增加通道容量，调整池化层以更好地保留竖直方向的字符特征
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 60x160 -> 30x80

            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 30x80 -> 15x40

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # 15x40 -> 7x40

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)) # 7x40 -> 3x40
        )

        # 强化序列学习能力：2 层 LSTM + Dropout
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2,
                           bidirectional=True, batch_first=True, dropout=0.25)
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.mean(dim=2).permute(0, 2, 1) # [Batch, Seq_len=40, Features=512]

        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)

        output = output.permute(1, 0, 2)
        return nn.functional.log_softmax(output, dim=2)