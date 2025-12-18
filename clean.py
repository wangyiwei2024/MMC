import cv2
import numpy as np
import os
from pathlib import Path

# ---------------------- 配置参数（已帮你填好路径） ----------------------
# 你的train_2k绝对路径（Windows格式）
DATA_ROOT = Path(r"/data/train_2k")
# color模态文件夹路径
MODAL_FOLDER = DATA_ROOT / "color"
# 噪声判断阈值（可根据实际调整，越小越严格）
NOISE_THRESHOLD = 1000
# 输出噪声文件名的txt路径（生成在脚本同目录）
OUTPUT_TXT = Path("ll.txt")


# ------------------------------------------------------------------------

def is_noise_image(image_path):
    """通过Canny边缘检测判断是否为噪声图（模糊/纯色/读取失败的图）"""
    # 读取图像（解决中文路径/特殊字符问题）
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return True  # 读取失败的图直接判定为噪声

    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Canny边缘检测（提取图像边缘）
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    # 计算边缘像素总和（值越小，图像越模糊）
    edge_sum = np.sum(edges)

    # 边缘总和小于阈值 → 判定为噪声
    return edge_sum < NOISE_THRESHOLD


def detect_noise_files():
    # 1. 验证路径是否存在
    if not DATA_ROOT.exists():
        print(f"❌ 错误：train_2k路径不存在 → {DATA_ROOT}")
        return
    if not MODAL_FOLDER.exists():
        print(f"❌ 错误：color文件夹不存在 → {MODAL_FOLDER}")
        return

    # 2. 统计color文件夹里的图片数量（匹配png/jpg/jpeg格式）
    img_files = list(MODAL_FOLDER.glob("*.png")) + list(MODAL_FOLDER.glob("*.jpg")) + list(MODAL_FOLDER.glob("*.jpeg"))
    print(f"✅ 成功找到color文件夹，共扫描到 {len(img_files)} 张图片")
    if len(img_files) == 0:
        print("❌ 警告：color文件夹里没有图片！")
        return

    # 3. 检测噪声图
    noise_filenames = []
    for img_file in img_files:
        if is_noise_image(img_file):
            noise_filenames.append(img_file.name)
            print(f"检测到噪声图：{img_file.name}")

    # 4. 写入ll.txt（解决中文文件名乱码问题）
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for fname in noise_filenames:
            f.write(f"{fname}\n")

    print(f"\n✅ 完成！共检测到 {len(noise_filenames)} 个噪声图，已写入 {OUTPUT_TXT}")


if __name__ == "__main__":
    detect_noise_files()