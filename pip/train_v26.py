from ultralytics import YOLO,settings
import os
import shutil
import torch
from datetime import datetime


def check_tensorboard():
    """检测并提醒 TensorBoard 状态"""
    try:
        import tensorboard
        print("✅ TensorBoard 已就绪。训练开始后，可通过命令查看: tensorboard --logdir runs/train")
    except ImportError:
        print("⚠️ 警告: 未检测到 TensorBoard 环境！")
        print("建议在终端运行: pip install tensorboard 以便开启可视化监控。")


def manage_checkpoints(project_dir, run_name):
    """
    智能管理断点：检查、恢复或清理损坏文件
    返回: resume_status (bool), model_path (str)
    """
    weights_dir = os.path.join(project_dir, run_name, 'weights')
    last_pt = os.path.join(weights_dir, 'last.pt')
    best_pt = os.path.join(weights_dir, 'best.pt')

    if not os.path.exists(last_pt):
        return False, 'models/yolo26n.pt'  # 没有断点，从预训练模型开始

    print(f"🔍 发现历史训练记录，正在检查断点完整性: {last_pt}")
    try:
        # 尝试加载检查点，看是否损坏
        ckpt = torch.load(last_pt, map_location='cpu')
        current_epoch = ckpt.get('epoch', -1)
        total_epochs = ckpt.get('args', {}).get('epochs', 250)

        if current_epoch >= total_epochs - 1:
            print("=" * 70)
            print(f"🎉 该任务已完成训练！最佳权重位于: {best_pt}")
            print("=" * 70)
            exit(0)  # 已完成则直接退出

        print(f"✅ 断点完好，准备从第 {current_epoch + 1} 轮恢复训练...")
        return True, last_pt

    except Exception as e:
        print(f"❌ 警告: last.pt 断点已损坏 ({str(e)})")
        # 自动备份/清理损坏文件
        corrupted_path = last_pt + f".corrupted_{datetime.now().strftime('%Y%m%d%H%M')}"
        shutil.move(last_pt, corrupted_path)
        print(f"♻️ 已将损坏的断点隔离至: {corrupted_path}")

        # 尝试使用 best.pt 抢救
        if os.path.exists(best_pt):
            print("💡 发现 best.pt，尝试从最佳验证点恢复训练...")
            return True, best_pt

        print("⚠️ 未找到可用的替代权重，将作为全新任务重新开始训练。")
        return False, 'yolo26n.pt'


def train_sewer_v26():
    """
    训练 v26 版本 - 管道缺陷检测 (集成断点与TensorBoard管理)
    """
    # 【新增：强制激活 TensorBoard 底层开关】
    settings.update({'tensorboard': True})
    os.environ['ULTRALYTICS_TENSORBOARD'] = 'True'
    
    # 基础路径配置
    PROJECT_DIR = 'runs/train'
    RUN_NAME = 'data_v26'  # 如果需要每次独立比对，可改为 f'data_v26_{datetime.now().strftime("%m%d_%H%M")}'




    # 1. 检测 TensorBoard
    check_tensorboard()

    # 2. 智能处理断点与模型初始化
    is_resume, start_model_path = manage_checkpoints(PROJECT_DIR, RUN_NAME)

    print(f"\n🚀 正在初始化 YOLOv26 模型 (载入: {start_model_path})...")
    model = YOLO(start_model_path)

    # 如果是恢复训练，直接调用 resume，无需再传一大堆超参数
    if is_resume:
        model.train(resume=True)
        return

    # 3. 全新训练的超参数配置
    cfg = dict(
        data='data/data.yaml',

        # 基础参数
        epochs=250,
        batch=4,
        imgsz=640,
        rect=True,
        device='0,1',
        workers=8,

        # 路径与日志管理
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,  # 允许覆盖/追加到当前目录，方便 TensorBoard 统一读取
        save_period=5,
        patience=30,
        plots=True,  # 强制生成可视化图表
        save=True,  # 强制保存权重与日志

        # 优化器
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,  # [优化建议] 延长 warmup 平稳适应管道特征
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        cos_lr=True,

        # 数据增强 (针对管道光照与水雾)
        mosaic=1.0,
        close_mosaic=30,  # [优化建议] 提前关闭 mosaic，强化真实分布学习
        mixup=0.1,
        copy_paste=0.1,
        scale=0.7,
        degrees=15.0,  # [优化建议] 模拟管道机器人爬行倾斜
        translate=0.1,
        hsv_h=0.015,
        hsv_s=0.8,  # [优化建议] 增强饱和度扰动
        hsv_v=0.6,  # [优化建议] 增强亮度扰动，对抗手电筒效应
        fliplr=0.5,

        # 损失函数与标签
        box=7.5,
        cls=2.5,
        dfl=2.0,  # [优化建议] 加大 DFL 权重，解决边界模糊
        label_smoothing=0.15,

        # 其他检测参数
        amp=True,
        val=True,
        cache='ram',
        iou=0.4,
        conf=0.001,
        max_det=500,
    )

    print(f"\n{'=' * 70}")
    print(f"🔥 YOLOv26 训练正式启动:")
    print(f"  模式: End-to-End (NMS-Free)")
    print(f"  日志: {os.path.join(PROJECT_DIR, RUN_NAME)}")
    print(f"{'=' * 70}")

    # 开始训练
    model.train(**cfg)

    # 验证
    print('\n📈 开始 v26 性能验证...')
    metrics = model.val(data=cfg['data'])

    print(f"\n📊 v26 验证结果 (End-to-End):")
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    train_sewer_v26()