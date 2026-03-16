import os
import shutil
from tqdm import tqdm

# --- 1. 核心映射配置 ---
# {新ID: 旧ID}
ID_MAPPING = {
    0: 1,  # 树根 -> SG
    1: 2,  # 沉积物 -> CJ
    2: 0,  # 裂缝 -> PL
    3: 3,  # 垃圾 -> ZW
    4: 7,  # 错口 -> CK
    5: 12  # 穿入 -> CR
}

# --- 2. 路径配置 ---
NEW_DATA_ROOT = "/home/daihui/yolov26-main/newdata"# 数据的解压位置
OLD_DATA_ROOT = "/home/daihui/yolov10-main/data"  # 原本的数据位置
OUTPUT_DIR = "/home/daihui/yolov26-main/data_merged"  # 合并后的新仓库


def merge_data():
    for split in ['train', 'val']:
        print(f"正在处理 {split} 数据集...")

        # 定义输入输出子目录
        src_img_dir = os.path.join(NEW_DATA_ROOT, "images")
        src_lbl_dir = os.path.join(NEW_DATA_ROOT, "labels")

        dst_img_dir = os.path.join(OUTPUT_DIR, "images", split)
        dst_lbl_dir = os.path.join(OUTPUT_DIR, "labels", split)

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        # 1. 首先拷贝原有的 15 类旧数据 (保持基准)
        old_img_src = os.path.join(OLD_DATA_ROOT, "images", split)
        old_lbl_src = os.path.join(OLD_DATA_ROOT, "labels", split)

        if os.path.exists(old_img_src):
            print(f"正在同步原有 {split} 基础数据...")
            for f in tqdm(os.listdir(old_img_src)):
                shutil.copy(os.path.join(old_img_src, f), os.path.join(dst_img_dir, f))
                # 对应拷贝标签
                lbl_f = f.rsplit('.', 1)[0] + ".txt"
                if os.path.exists(os.path.join(old_lbl_src, lbl_f)):
                    shutil.copy(os.path.join(old_lbl_src, lbl_f), os.path.join(dst_lbl_dir, lbl_f))

        # 2. 处理并重映射新买的 6 类数据
        print(f"正在重映射并注入新购 {split} 数据...")
        new_label_files = [f for f in os.listdir(src_lbl_dir) if f.endswith('.txt')]

        for lbl_name in tqdm(new_label_files):
            # 处理标签重映射
            new_lines = []
            with open(os.path.join(src_lbl_dir, lbl_name), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    new_id = int(parts[0])
                    if new_id in ID_MAPPING:
                        parts[0] = str(ID_MAPPING[new_id])  # 执行翻译
                        new_lines.append(" ".join(parts))

            if new_lines:
                # 写入新路径 (为防止文件名冲突，加个前缀)
                save_name = f"new_{lbl_name}"
                with open(os.path.join(dst_lbl_dir, save_name), 'w') as f:
                    f.write("\n".join(new_lines))

                # 拷贝对应图片
                img_name_base = lbl_name.rsplit('.', 1)[0]
                found_img = False
                for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
                    potential_img = os.path.join(src_img_dir, img_name_base + ext)
                    if os.path.exists(potential_img):
                        shutil.copy(potential_img, os.path.join(dst_img_dir, f"new_purchase_{img_name_base}{ext}"))
                        found_img = True
                        break

    print(f"\n✅ 数据合并完成！总目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    merge_data()