import os
import glob
from collections import Counter, defaultdict
import yaml

def check_labels(label_dir, data_yaml=None):
    """
    检查YOLO格式标签文件
    Args:
        label_dir: 标签文件夹路径
        data_yaml: 数据集配置文件路径，用于验证类别名称
    """
    files = glob.glob(os.path.join(label_dir, "*.txt"))
    if not files:
        print(f"警告：在 {label_dir} 中未找到任何 .txt 文件")
        return

    # 加载类别名称（如果提供data.yaml）
    class_names = None
    if data_yaml and os.path.exists(data_yaml):
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
            class_names = data_cfg.get('names', [])
            print(f"类别名称（共 {len(class_names)} 类）: {class_names}")

    # 统计变量
    class_counts = Counter()           # 总实例数
    image_class_counts = defaultdict(set)  # 每个图像包含的类别ID（集合去重）
    invalid_files = []                  # 格式错误的文件
    empty_files = []                     # 空文件
    coord_out_range = []                 # 坐标超出[0,1]的文件

    for f in files:
        img_id = os.path.basename(f)    # 用于记录
        has_any_valid = False
        try:
            with open(f, 'r') as file:
                lines = file.readlines()
                if not lines:
                    empty_files.append(img_id)
                    continue

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue  # 跳过空行
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"警告：文件 {img_id} 第 {line_num} 行格式错误（应有5个字段）")
                        invalid_files.append(img_id)
                        continue
                    try:
                        cls_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                    except ValueError:
                        print(f"警告：文件 {img_id} 第 {line_num} 行无法解析数字")
                        invalid_files.append(img_id)
                        continue

                    # 检查类别ID范围
                    if class_names and (cls_id < 0 or cls_id >= len(class_names)):
                        print(f"警告：文件 {img_id} 类别ID {cls_id} 超出配置范围")

                    # 检查坐标是否归一化 (0~1)
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        coord_out_range.append(img_id)

                    class_counts[cls_id] += 1
                    image_class_counts[img_id].add(cls_id)
                    has_any_valid = True

        except Exception as e:
            print(f"错误：无法读取文件 {img_id} - {e}")
            invalid_files.append(img_id)

        if not has_any_valid and f not in empty_files:
            empty_files.append(img_id)

    # 打印统计结果
    print("\n" + "="*50)
    print("📊 数据集标签统计")
    print("="*50)
    print(f"📁 标签文件总数: {len(files)}")
    print(f"✅ 有效文件: {len(files) - len(invalid_files) - len(empty_files)}")
    print(f"⚠️  空文件: {len(empty_files)}")
    print(f"❌ 格式错误文件: {len(invalid_files)}")
    print(f"📏 坐标超出范围的文件: {len(coord_out_range)}")

    # 按实例数排序输出
    print("\n🔢 各类别实例数量（总边界框数）:")
    for cls_id, count in sorted(class_counts.items()):
        cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"ID:{cls_id}"
        print(f"  {cls_name:20} : {count:6}")

    # 图像级统计（至少包含该类别的一张图像）
    print("\n🖼️  各类别图像数量（至少包含一个实例）:")
    image_counts = Counter()
    for img_id, cls_set in image_class_counts.items():
        for cls_id in cls_set:
            image_counts[cls_id] += 1
    for cls_id, count in sorted(image_counts.items()):
        cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"ID:{cls_id}"
        print(f"  {cls_name:20} : {count:6}")

    # 类别平衡性提示
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        if min_count > 0 and max_count / min_count > 10:
            print("\n⚠️  类别严重不平衡！最大/最小实例数比 > 10，建议考虑类别重采样或调整损失权重。")
    print("="*50)

if __name__ == "__main__":
    LABEL_DIR = "/home/daihui/yolov26-main/data/labels/train"
    DATA_YAML = "/home/daihui/yolov26-main/runs/detect/runs/train/data_v26/weights/data.yaml"  # 根据实际路径修改
    check_labels(LABEL_DIR, DATA_YAML)