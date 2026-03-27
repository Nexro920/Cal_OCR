from captcha.image import ImageCaptcha
import random
import os
import csv
import multiprocessing

save_dir = r"math_captchas"
train_dir = os.path.join(save_dir, "train")
val_dir = os.path.join(save_dir, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

captcha_generator = ImageCaptcha(width=160, height=60)

def generate_single_image(args):
    i, total_images, train_ratio, t_dir, v_dir = args

    operator = random.choice(['+', '-', '*', '/'])

    if operator == '/':
        num2 = random.randint(1, 30)
        expected_answer = random.randint(1, 15)
        num1 = num2 * expected_answer
        text = f"{num1}/{num2}="
        answer = expected_answer
    else:
        num1 = random.randint(1, 99)
        num2 = random.randint(1, 99)
        if operator == '-' and num1 < num2:
            num1, num2 = num2, num1

        text = f"{num1}{operator}{num2}="
        # 这里仅用于生成标签答案，生成阶段是安全的
        answer = int(eval(f"{num1}{operator}{num2}"))

    filename = f"{i:06d}.png"

    if i < total_images * train_ratio:
        filepath = os.path.join(t_dir, filename)
        split_type = "train"
    else:
        filepath = os.path.join(v_dir, filename)
        split_type = "val"

    captcha_generator.write(text, filepath)

    # 直接返回变长 text 作为 label
    return {"filename": filename, "label": text, "answer": answer, "split": split_type}

if __name__ == '__main__':
    total_images = 10000
    train_ratio = 0.8
    print("开始多进程生成变长验证码数据集...")

    pool_args = [(i, total_images, train_ratio, train_dir, val_dir) for i in range(total_images)]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(generate_single_image, pool_args)

    train_data = [r for r in results if r['split'] == 'train']
    val_data = [r for r in results if r['split'] == 'val']

    for split_name, data in [("train", train_data), ("val", val_data)]:
        csv_path = os.path.join(save_dir, f"{split_name}_labels.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label", "answer", "split"])
            writer.writeheader()
            writer.writerows(data)

    print(f"生成完毕！共生成 {total_images} 张图片。标签为真实长度，完美适配 CRNN + CTC。")