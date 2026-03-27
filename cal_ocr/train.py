import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

from .ocr_dataset import get_dataloader, VOCAB, IDX2CHAR
from .ocr_model import CRNN

def decode_predictions(preds):
    """ CTC 贪心解码：合并重复项，去除空白符 """
    _, max_indices = torch.max(preds, dim=2) # [T, Batch]
    max_indices = max_indices.permute(1, 0).cpu().numpy() # [Batch, T]

    decoded_strings = []
    for seq in max_indices:
        char_list = []
        for i in range(len(seq)):
            if seq[i] != 0 and (not (i > 0 and seq[i - 1] == seq[i])): # 0 是 Blank
                char_list.append(IDX2CHAR[seq[i]])
        decoded_strings.append(''.join(char_list))
    return decoded_strings

def train_model(num_epochs=50, batch_size=128, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 当前使用设备: {device}")
    os.makedirs("weights", exist_ok=True)

    # 验证集不应 drop_last，保证评估准确性。启用多进程加载加速。
    train_loader = get_dataloader(csv_file="math_captchas/train_labels.csv", img_dir="math_captchas/train", batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = get_dataloader(csv_file="math_captchas/val_labels.csv", img_dir="math_captchas/val", batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    model = CRNN(num_classes=len(VOCAB)).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # 修复：确保 scaler 只在 cuda 下完全启用
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for images, targets, target_lengths in pbar:
            images = images.to(device)
            targets = targets.to(device)
            batch_s = images.size(0)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                preds = model(images)
                input_lengths = torch.full(size=(batch_s,), fill_value=preds.size(0), dtype=torch.long)
                loss = criterion(preds, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.5f}"})

        scheduler.step()

        # 验证阶段
        model.eval()
        correct_sentences, total_sentences = 0, 0

        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                images = images.to(device)
                preds = model(images)

                decoded_preds = decode_predictions(preds)

                targets = targets.cpu().numpy()
                target_lengths = target_lengths.cpu().numpy()
                for i in range(len(decoded_preds)):
                    true_seq = targets[i][:target_lengths[i]]
                    true_str = ''.join([IDX2CHAR[c] for c in true_seq])
                    if decoded_preds[i] == true_str:
                        correct_sentences += 1
                total_sentences += len(decoded_preds)

        val_acc = correct_sentences / total_sentences
        print(f"Epoch {epoch+1} -> Train Loss: {train_loss/len(train_loader):.4f} | Val Sentence Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "weights/best_crnn_model.pth")
            print(f"🔥 新纪录！模型已保存！")

if __name__ == '__main__':
    train_model(num_epochs=30, batch_size=128, learning_rate=0.001)