#!/usr/bin/env python3
"""
合并评估脚本：
- 优先使用项目自带的 tools/eval.py（如果存在且支持 --classes 参数）
- 否则使用 Ultralytics API 进行手动评估
- 仅评估指定的类别（旧ID：0,1,2,3,7,12）
"""

import subprocess
import sys
import os
from pathlib import Path
# 建立 ID 到中文的映射（根据你提供的列表）
ID_TO_CHINESE = {
    0: "PL-破裂",
    1: "SG-树根",
    2: "CJ-沉积",
    3: "ZW-障碍物",
    4: "CK-错口",
    5: "CR-异物穿入",
    6: "AJ-支管暗接",
    7: "BX-变形",
    8: "FS-腐蚀",
    9: "QF-起伏",
    10: "SL-渗漏",
    11: "TJ-脱节",
    12: "TL-接口材料脱落",
    13: "CQ-残墙坝根",
    14: "FZ-浮渣",
    15: "JG-结垢"
}
# ========== 硬编码配置参数 ==========
MODEL_PATH = "/home/daihui/yolov26-main/runs/detect/runs/train/best.pt"   # 模型权重
DATA_YAML = "//home/daihui/yolov26-main/data/data.yaml"                # 数据集配置（含所有13类）
CLASSES = [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15]                   # 要评估的类别（旧ID）
BATCH_SIZE = 8
IMGSZ = 640
CONF = 0.001
IOU = 0.65
DEVICE = "0"                                     # GPU ID 或 "cpu"
# 以下为 eval.py 专用参数（若使用）
CONF_THRES = CONF
IOU_THRES = IOU
# ===================================

def use_eval_py():
    """尝试使用 tools/eval.py 进行评估"""
    script_dir = Path(__file__).parent.absolute()
    eval_script = script_dir / 'tools' / 'eval.py'
    if not eval_script.exists():
        print("[信息] 未找到 tools/eval.py，将使用 Ultralytics API 评估。")
        return False

    print("[信息] 找到 tools/eval.py，尝试使用它进行评估...")
    # 构建命令
    cmd = [
        sys.executable, str(eval_script),
        '--data', DATA_YAML,
        '--weights', MODEL_PATH,
        '--batch-size', str(BATCH_SIZE),
        '--img-size', str(IMGSZ),
        '--conf-threshold', str(CONF_THRES),
        '--iou-threshold', str(IOU_THRES),
        '--device', DEVICE,
        '--classes', *[str(c) for c in CLASSES]
    ]
    print("执行命令: " + ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[错误] eval.py 执行失败: {e}")
        print("[信息] 将回退到 Ultralytics API 评估。")
        return False
    except FileNotFoundError:
        print("[错误] 未找到 Python 解释器，请检查环境。")
        return False


def use_ultralytics():
    """使用 Ultralytics YOLO API 进行评估并输出中文结果"""
    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        print("[错误] 未安装 ultralytics 库")
        sys.exit(1)

    print("[信息] 正在加载模型并开始中文指标评估...")
    device = DEVICE if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)

    # 运行评估
    results = model.val(
        data=DATA_YAML,
        classes=CLASSES,
        batch=BATCH_SIZE,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=device,
        plots=True,
        save_json=True,
    )

    print("\n" + "=" * 50)
    print(f"{'类别名称':<15} | {'Instances':<10} | {'P':<8} | {'R':<8} | {'mAP50':<8}")
    print("-" * 50)

    # 提取所有类别的指标
    # results.box.ap_class_index 是当前评估中包含的类别索引
    for i, class_idx in enumerate(results.box.ap_class_index):
        # 获取中文名，如果不在字典里则显示原始 ID
        name = ID_TO_CHINESE.get(class_idx, f"ID-{class_idx}")

        precision = results.box.p[i]  # Precision
        recall = results.box.r[i]  # Recall
        map50 = results.box.ap50[i]  # mAP50

        # 统计当前类别的实例数量 (从 results.results_dict 获取)
        # 注意：不同版本 ultralytics 结构略有不同，这里取各类别 AP 结果
        print(f"{name:<15} | {'-':<10} | {precision:<8.3f} | {recall:<8.3f} | {map50:<8.3f}")

    print("-" * 50)
    print(
        f"{'综合平均 (all)':<15} | {'-':<10} | {results.box.mp:<8.3f} | {results.box.mr:<8.3f} | {results.box.map50:<8.3f}")
    print("=" * 50)

    print(f"\n[提示] 详细图表已保存至: {results.save_dir}")

def main():
    # 检查必要文件
    if not os.path.isfile(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.isfile(DATA_YAML):
        print(f"[错误] 数据集配置文件不存在: {DATA_YAML}")
        sys.exit(1)

    # 优先使用 eval.py
    if not use_eval_py():
        use_ultralytics()

if __name__ == "__main__":
    main()