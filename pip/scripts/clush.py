import fiftyone as fo
from fiftyone import ViewField as F  # 修正这里
import numpy as np
# 1. 数据集初始化 (保持不变)
if "combined_dataset" in fo.list_datasets():
    fo.delete_dataset("combined_dataset")

dataset = fo.Dataset("combined_dataset")

# 2. 导入数据
dataset.add_dir(
    dataset_dir="/home/daihui/yolov26-main/data",
    dataset_type=fo.types.YOLOv5Dataset,
    tags="old_data"
)
dataset.add_dir(
    dataset_dir="/home/daihui/yolov26-main/PIPE_test",
    dataset_type=fo.types.YOLOv5Dataset,
    tags="new_data"
)

# 3. 计算元数据（获取图片宽高）
dataset.compute_metadata()

# 4. 批量计算框的大小（这种写法比循环更快）
for sample in dataset:
    if sample.ground_truth:
        img_w = sample.metadata.width
        img_h = sample.metadata.height
        for det in sample.ground_truth.detections:
            # FiftyOne 存储: [x, y, width, height] (均为相对比例)
            det["width_px"] = det.bounding_box[2] * img_w
            det["height_px"] = det.bounding_box[3] * img_h
            det["area_px"] = det["width_px"] * det["height_px"]
        sample.save()

# 5. 启动 App
session = fo.launch_app(dataset, address="0.0.0.0", port=5151)
session.wait()
