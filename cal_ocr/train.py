import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from pathlib import Path
import Levenshtein

from .ocr_dataset import get_dataloader, VOCAB
from .ocr_model import CRNN
from .predict import decode_predictions

def calculate_cer(pred_str, true_str):
    if len(true_str) == 0:
        return 1.0
    return Levenshtein.distance(pred_str, true_str) / len(true_str)

def train_model(num_epochs=80, batch_size=128, learning_rate=0.001):  # ← 调低
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 当前设备: {device} | 准备起飞！")

    # 【修复】正确的项目根目录
    project_root = Path(__file__).resolve().parents[1]   # Cal_OCR/
    data_dir = project_root / "scripts/math2"

    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(exist_ok=True)
    best_path = weights_dir / "best_crnn_model.pth"

    print("📦 加载数据并开启在线增强...")
    train_loader = get_dataloader(
        str(data_dir / "train_labels.csv"), str(data_dir / "train"),
        batch_size=batch_size, shuffle=True, augment=True
    )
    val_loader = get_dataloader(
        str(data_dir / "val_labels.csv"), str(data_dir / "val"),
        batch_size=batch_size, shuffle=False, augment=False, drop_last=False  # ← 关键修复
    )

    model = CRNN(num_classes=len(VOCAB), image_channels=3).to(device)

    if best_path.exists():
        print(f"🔄 加载历史最佳权重: {best_path}")
        state = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # 使用 AdamW + OneCycleLR，收敛速度起飞
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for images, targets_1d, target_lengths in pbar:
            images = images.to(device)
            targets_1d = targets_1d.to(device)
            target_lengths = target_lengths.to(device)          # ← 关键修复！
            batch_s = images.size(0)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                preds = model(images)
                # 修复 input_lengths 未放在指定 device 的性能损耗
                input_lengths = torch.full((batch_s,), preds.size(0), dtype=torch.long, device=device)
                loss = criterion(preds, targets_1d, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.5f}"})

        # 验证环节（保持不变，仅优化可读性）
        model.eval()
        correct, total, total_cer = 0, 0, 0.0
        with torch.no_grad():
            for images, targets_1d, target_lengths in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val  ]", leave=False):
                images = images.to(device)
                preds = model(images)
                decoded_preds = decode_predictions(preds)

                targets_1d = targets_1d.cpu().numpy()
                target_lengths = target_lengths.cpu().numpy()

                # 手动切分 1D targets 进行对比
                ptr = 0
                for i, pred_str in enumerate(decoded_preds):
                    length = target_lengths[i]
                    true_seq = targets_1d[ptr:ptr + length]
                    ptr += length
                    true_str = ''.join(VOCAB[c] for c in true_seq)

                    if pred_str == true_str:
                        correct += 1
                    total_cer += calculate_cer(pred_str, true_str)
                    total += 1

        val_acc = correct / total
        val_char_acc = max(0.0, 1 - total_cer / total)
        print(f"Epoch {epoch+1} → Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Char Acc: {val_char_acc*100:.2f}% | Val Sent Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"🔥 新最佳模型已保存: {best_path}")

if __name__ == '__main__':
    train_model(num_epochs=80, batch_size=128)
