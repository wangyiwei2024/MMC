import os
from pathlib import Path

# ---------------------- 配置路径（必须核对！） ----------------------
# 1. 噪声图片名列表：ll.txt 的路径
LL_TXT_PATH = Path(r"D:\打榜大作业final\ll.txt")
# 2. 要清理的标签文件：train_labels.txt 的路径
TRAIN_LABELS_PATH = Path(r"D:\打榜大作业final\data\train_2k\train_labels.txt")


# -------------------------------------------------------------------

def load_noise_filenames(ll_txt_path):
    """从 ll.txt 加载所有噪声图片名（去重、去空）"""
    if not ll_txt_path.exists():
        raise FileNotFoundError(f"❌ 找不到 ll.txt！路径：{ll_txt_path}")

    with open(ll_txt_path, "r", encoding="utf-8") as f:
        # 读取所有行，去除空行、换行符，确保每个元素是纯净的图片名
        noise_names = [line.strip() for line in f if line.strip()]

    # 去重（避免重复删除）
    noise_names = list(set(noise_names))
    print(f"✅ 从 ll.txt 加载到 {len(noise_names)} 个噪声图片名")
    return noise_names


def clean_train_labels(noise_names, train_labels_path):
    """清理 train_labels.txt：删除包含噪声图片名的行"""
    # 1. 备份原标签文件（防止误删，备份文件会加 .bak 后缀）
    backup_path = train_labels_path.with_suffix(".txt.bak")
    if not backup_path.exists():
        # 复制原文件到备份
        with open(train_labels_path, "r", encoding="utf-8") as f_src, \
                open(backup_path, "w", encoding="utf-8") as f_dst:
            f_dst.write(f_src.read())
        print(f"✅ 已创建备份文件：{backup_path}")

    # 2. 读取原标签文件内容
    with open(train_labels_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    # 3. 过滤：保留不包含任何噪声图片名的行
    kept_lines = []
    deleted_count = 0
    for line in all_lines:
        line_stripped = line.strip()
        # 判断当前行是否包含任意一个噪声图片名（核心逻辑）
        is_noise = any(noise_name in line_stripped for noise_name in noise_names)
        if is_noise:
            deleted_count += 1
            print(f"❌ 已删除：{line_stripped}")  # 打印删除的条目，方便核对
        else:
            kept_lines.append(line)  # 保留非噪声行

    # 4. 将过滤后的内容写回原标签文件
    with open(train_labels_path, "w", encoding="utf-8") as f:
        f.writelines(kept_lines)

    # 5. 输出最终统计结果
    print(f"\n📊 清理完成！")
    print(f"   - 原条目总数：{len(all_lines)}")
    print(f"   - 删除条目数：{deleted_count}")
    print(f"   - 剩余条目数：{len(kept_lines)}")
    print(f"   - 备份文件：{backup_path}")


if __name__ == "__main__":
    try:
        # 安全确认（防止误操作）
        confirm = input("⚠️ 警告：此操作会修改 train_labels.txt 并创建备份！请确认是否继续？(输入 yes 确认)：")
        if confirm.lower() != "yes":
            print("🚫 操作已取消")
            exit()

        # 执行核心逻辑
        noise_list = load_noise_filenames(LL_TXT_PATH)
        clean_train_labels(noise_list, TRAIN_LABELS_PATH)

    except Exception as e:
        print(f"\n❌ 程序出错：{str(e)}")