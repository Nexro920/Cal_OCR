from captcha.image import ImageCaptcha
import random
import os
import csv
import multiprocessing
import cv2
import numpy as np
from pathlib import Path

save_dir = str(Path(__file__).parent.parent / "math_captchas")
train_dir = str(Path(save_dir) / "train")
val_dir = str(Path(save_dir) / "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def add_extreme_augmentation(img_array):
    h, w = img_array.shape[:2]

    if random.random() < 0.6:
        tx, ty = random.randint(1, 3), random.randint(1, 3)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        shadow = cv2.warpAffine(img_array, M, (w, h), borderValue=(255, 255, 255))
        img_array = cv2.multiply(img_array.astype(float)/255.0, shadow.astype(float)/255.0)
        img_array = (img_array * 255).astype(np.uint8)

    if random.random() < 0.5:
        gradient = np.zeros((h, w, 3), dtype=np.uint8)
        c1 = np.random.randint(100, 255, size=3)
        c2 = np.random.randint(100, 255, size=3)
        for i in range(w):
            alpha = i / w
            gradient[:, i] = c1 * (1 - alpha) + c2 * alpha

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)

        img_bg = cv2.bitwise_and(gradient, gradient, mask=mask_inv)
        img_fg = cv2.bitwise_and(img_array, img_array, mask=mask)
        img_array = cv2.add(img_bg, img_fg)

    if random.random() < 0.7:
        thickness = random.randint(1, 4)
        color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        cv2.rectangle(img_array, (0, 0), (w-1, h-1), color, thickness)

    if random.random() < 0.4:
        amp = random.uniform(1.5, 3.5)
        freq = random.uniform(10, 25)
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        for i in range(h):
            for j in range(w):
                map_x[i, j] = j + amp * np.sin(i / freq)
                map_y[i, j] = i + amp * np.cos(j / freq)
        img_array = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return img_array

def generate_single_image(args):
    # 修改点：在子进程内部初始化，解决 Windows Multiprocessing PicklingError
    captcha_generator = ImageCaptcha(width=160, height=60)

    i, total_images = args

    operators = ['+', '-', '*', '/']
    suffixes = ['', '=', '=?']

    operator = operators[(i // 3) % 4]
    suffix = suffixes[i % 3]

    if operator == '/':
        num2 = random.randint(1, 30)
        expected_answer = random.randint(1, 15)
        num1 = num2 * expected_answer
        text = f"{num1}/{num2}{suffix}"
        answer = expected_answer
    else:
        num1 = random.randint(1, 99)
        num2 = random.randint(1, 99)
        if operator == '-' and num1 < num2:
            num1, num2 = num2, num1

        text = f"{num1}{operator}{num2}{suffix}"
        answer = int(eval(f"{num1}{operator}{num2}"))

    filename = f"{i:06d}.png"

    image_pil = captcha_generator.generate_image(text)
    img_array = np.array(image_pil)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    img_augmented = add_extreme_augmentation(img_array)

    # 修改点：直接返回内存中的图片阵列，交给主进程统一洗牌保存
    return {"filename": filename, "label": text, "answer": answer, "image": img_augmented}

if __name__ == '__main__':
    total_images = 50000
    train_ratio = 0.8
    print("开始多进程生成【重度污染版】变长验证码数据集...")

    pool_args = [(i, total_images) for i in range(total_images)]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(generate_single_image, pool_args)

    # 修改点：生成全量数据字典后，进行完美随机打乱，消除分布偏差
    random.shuffle(results)
    split_idx = int(total_images * train_ratio)

    train_data = []
    val_data = []

    print("数据生成完毕，正在切分并写入磁盘...")
    for idx, r in enumerate(results):
        if idx < split_idx:
            cv2.imwrite(os.path.join(train_dir, r["filename"]), r["image"])
            train_data.append({"filename": r["filename"], "label": r["label"], "answer": r["answer"], "split": "train"})
        else:
            cv2.imwrite(os.path.join(val_dir, r["filename"]), r["image"])
            val_data.append({"filename": r["filename"], "label": r["label"], "answer": r["answer"], "split": "val"})

    for split_name, data in [("train", train_data), ("val", val_data)]:
        csv_path = os.path.join(save_dir, f"{split_name}_labels.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label", "answer", "split"])
            writer.writeheader()
            writer.writerows(data)

    print(f"生成完毕！共生成 {total_images} 张高难度抗干扰图片。")