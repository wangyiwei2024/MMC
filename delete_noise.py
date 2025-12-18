import os
from pathlib import Path

# ---------------------- 配置参数（根据你的实际路径修改） ----------------------
# 噪声文件名列表的txt路径（ll.txt）
NOISE_TXT_PATH = Path(r"D:\打榜大作业final\ll.txt")
# 数据集根路径（train_2k）
DATA_ROOT = Path(r"D:\打榜大作业final\data\train_2k")
# 三种模态文件夹名称（color/depth/infrared）
MODAL_FOLDERS = ["color", "depth", "infrared"]
# 图片后缀（你的文件都是.png）
IMG_SUFFIXES = [".png"]


# ------------------------------------------------------------------------

def extract_core_id(filename):
    """
    适配你的真实文件名格式：
    输入：000002_080_00000048_0_9.png → 输出：000002_080_00000048_0_9
    核心逻辑：直接去掉.png后缀，保留完整核心标识（无多余前缀）
    """
    # 仅去掉.png后缀，保留全部字符作为核心标识
    core_id = os.path.splitext(filename)[0]
    return core_id


def delete_noise_files():
    # 1. 校验路径是否存在
    if not NOISE_TXT_PATH.exists():
        print(f"❌ 错误：噪声列表文件不存在 → {NOISE_TXT_PATH}")
        return
    if not DATA_ROOT.exists():
        print(f"❌ 错误：数据集根路径不存在 → {DATA_ROOT}")
        return

    # 2. 读取ll.txt中的噪声文件名（需确保ll.txt里是完整的文件名，如000002_080_00000048_0_9.png）
    with open(NOISE_TXT_PATH, "r", encoding="utf-8") as f:
        noise_filenames = [line.strip() for line in f if line.strip()]
    if not noise_filenames:
        print("❌ 错误：ll.txt中无噪声文件名！")
        return

    # 3. 提取噪声核心标识（去掉后缀）
    noise_core_ids = [extract_core_id(fname) for fname in noise_filenames]
    print(f"✅ 读取到 {len(noise_core_ids)} 个噪声核心标识")

    # 4. 遍历三种模态文件夹，删除对应文件
    deleted_count = 0
    for modal in MODAL_FOLDERS:
        modal_path = DATA_ROOT / modal
        if not modal_path.exists():
            print(f"⚠️ 警告：模态文件夹不存在 → {modal_path}，跳过")
            continue

        # 遍历当前模态下的所有.png文件
        for img_file in modal_path.iterdir():
            if img_file.suffix not in IMG_SUFFIXES:
                continue  # 跳过非png文件

            # 提取当前文件的核心标识（去掉.png）
            file_core_id = extract_core_id(img_file.name)
            # 匹配噪声标识则删除
            if file_core_id in noise_core_ids:
                try:
                    img_file.unlink()  # 永久删除文件
                    deleted_count += 1
                    print(f"✅ 已删除：{img_file}")
                except Exception as e:
                    print(f"❌ 删除失败：{img_file} → 原因：{e}")

    # 5. 输出删除结果统计
    print(f"\n📊 删除完成！共删除 {deleted_count} 个噪声文件")
    print(f"📋 涉及模态文件夹：{MODAL_FOLDERS}")
    print(f"📝 噪声源文件：{NOISE_TXT_PATH}")


if __name__ == "__main__":
    # 安全二次确认（防止误删）
    confirm = input("⚠️ 警告：此操作将永久删除文件！请确认是否继续？(输入yes确认)：")
    if confirm.lower() == "yes":
        delete_noise_files()
    else:
        print("🚫 操作已取消")