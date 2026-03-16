import fiftyone as fo
import numpy as np
from PIL import Image

# 1. 加载数据集
dataset = fo.load_dataset("combined_dataset")

# 2. 强制注册并注入亮度字段 (PRIMITIVES)
print("正在计算亮度并强制注册字段...")
filepaths = dataset.values("filepath")
brights = [float(np.array(Image.open(fp).convert("L")).mean()) for fp in filepaths]

# 使用 set_values 会自动创建字段
dataset.set_values("mean_intensity", brights)

# 3. 计算面积 (LABELS 内部)
print("正在注入面积数据...")
dataset.map_labels(
    "ground_truth",
    lambda det: det.update({"area": det.bounding_box[2] * det.bounding_box[3]})
)

# 4. 关键步骤：强制保存修改并同步 Schema
dataset.save()
dataset.reload()

print("字段注入完成！正在启动 App...")

# 5. 启动 App
session = fo.launch_app(dataset, address="0.0.0.0", port=5151)
session.wait()