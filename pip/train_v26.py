import cv2
from ultralytics import YOLO, settings
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import shutil
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def check_tensorboard():
    """检测并提醒 TensorBoard 状态"""
    try:
        import tensorboard
        print("TensorBoard 已就绪。训练开始后，可通过命令查看: tensorboard --logdir runs/train")
    except ImportError:
        print("警告: 未检测到 TensorBoard 环境！")
        print("建议在终端运行: pip install tensorboard 以便开启可视化监控。")


def get_latest_run_dir(project_dir, base_name):
    """查找指定前缀的最新训练目录"""
    search_pattern = os.path.join(project_dir, f"{base_name}*")
    dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]
    if not dirs:
        return None
    # 按文件夹的修改时间倒序排列，取最新修改的一个
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[0]


def manage_checkpoints(project_dir, base_name):
    """
    智能管理断点：查找最新目录，检查恢复或清理损坏文件
    返回: is_resume (bool), model_path (str), current_run_name (str)
    """
    latest_dir = get_latest_run_dir(project_dir, base_name)

    # 预设一个全新的时间戳 RUN_NAME，以备不时之需
    timestamp = datetime.now().strftime('%m%d_%H%M')
    new_run_name = f"{base_name}_{timestamp}"

    if not latest_dir:
        return False, 'yolo26m.pt', new_run_name

    weights_dir = os.path.join(latest_dir, 'weights')
    last_pt = os.path.join(weights_dir, 'last.pt')
    best_pt = os.path.join(weights_dir, 'best.pt')
    latest_run_name = os.path.basename(latest_dir)

    if not os.path.exists(last_pt):
        # 存在文件夹但没权重（可能刚启动就手动掐断了），算作新任务
        return False, 'yolo26m.pt', new_run_name

    print(f"发现最新历史训练记录 [{latest_run_name}]，正在检查断点完整性...")
    try:
        ckpt = torch.load(last_pt, map_location='cpu')
        current_epoch = ckpt.get('epoch', -1)
        total_epochs = ckpt.get('args', {}).get('epochs', 250)

        if current_epoch >= total_epochs - 1:
            print("=" * 70)
            print(f"该任务已完成训练！最佳权重位于: {best_pt}")
            print("=" * 70)
            exit(0)

        print(f"断点完好，准备从第 {current_epoch + 1} 轮恢复训练...")
        # 恢复训练时，返回旧的目录名，YOLO会自动在旧目录续写日志
        return True, last_pt, latest_run_name

    except Exception as e:
        print(f"警告: last.pt 断点已损坏 ({str(e)})")
        corrupted_path = last_pt + f".corrupted_{datetime.now().strftime('%Y%m%d%H%M')}"
        shutil.move(last_pt, corrupted_path)
        print(f"已将损坏的断点隔离至: {corrupted_path}")

        if os.path.exists(best_pt):
            print("发现 best.pt，尝试从最佳验证点恢复训练...")
            return True, best_pt, latest_run_name

        print("未找到可用的替代权重，将生成新目录重新开始训练。")
        return False, 'yolo26m.pt', new_run_name


def train_sewer_v26():
    """训练 v26 版本 - 管道缺陷检测"""
    settings.update({'tensorboard': True})
    os.environ['ULTRALYTICS_TENSORBOARD'] = 'True'

    PROJECT_DIR = 'runs/train'
    BASE_NAME = 'data_v26-1'  # 你的基础前缀

    check_tensorboard()

    # 动态获取模型路径和本次训练要使用的文件夹名称
    is_resume, start_model_path, current_run_name = manage_checkpoints(PROJECT_DIR, BASE_NAME)

    print(f"\n正在初始化 YOLOv26 模型 (载入: {start_model_path})...")
    model = YOLO(start_model_path)

    if is_resume:
        # 恢复训练时 Ultralytics 会自动读取原来的 args，直接 run 即可
        model.train(resume=True)
        return

    # 全新训练的超参数配置
    cfg = dict(
        data='merged_dataset/data.yaml',

        epochs=250,
        batch=16,
        imgsz=640,
        rect=False,
        device='0,1',
        workers=8,

        project=PROJECT_DIR,
        name=current_run_name,
        exist_ok=True,
        save_period=5,
        patience=50,
        plots=True,
        save=True,

        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        cos_lr=True,

        #  数据增强
        mosaic=1.0,  # 保持高强度 mosaic 以处理小目标缺陷
        close_mosaic=20,  # 最后 20 轮关闭 mosaic 以精细化收敛
        mixup=0.1,
        copy_paste=0.1,  # 增加物体复制，解决样本不平衡
        scale=0.5,
        degrees=10.0,
        translate=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        fliplr=0.5,

        #  核心优化
        box=7.5,
        cls=1.5,
        dfl=2.0,
        label_smoothing=0.005,
        dropout=0.1,

        #  检测控制
        amp=True,
        val=True,
        cache='ram',
        iou=0.4,
        max_det=100,
    )

    print(f"\n{'=' * 70}")
    print(f"YOLOv26 训练正式启动:")
    print(f"  模式: {'断点续训' if is_resume else '全新训练'}")
    print(f"  日志目录: {os.path.join(PROJECT_DIR, current_run_name)}")
    print(f"{'=' * 70}")

    def on_train_epoch_end(trainer):
        """自定义回调：将 YOLO 保存的图片推送到 TensorBoard"""
        tb_writer = None
        # 寻找 TensorBoard 的 Writer 实例
        for logger in trainer.loggers.values():
            if isinstance(logger, SummaryWriter):
                tb_writer = logger
                break
                
        if tb_writer:
            # 在第 1 轮结束时，抓取训练集 batch 图像（通常只生成一次）
            if trainer.epoch == 0:
                img_path = os.path.join(trainer.save_dir, 'train_batch0.jpg')
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV 默认 BGR，转为 RGB
                    tb_writer.add_image('Training Data/Batch 0', img, trainer.epoch, dataformats='HWC')
                    
            # 定期（如每 10 轮）抓取验证集预测图像
            if trainer.epoch % 10 == 0:
                pred_path = os.path.join(trainer.save_dir, f'val_batch0_pred.jpg')
                if os.path.exists(pred_path):
                    img = cv2.imread(pred_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tb_writer.add_image('Validation/Predictions', img, trainer.epoch, dataformats='HWC')

    # 注册回调
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    model.train(**cfg)

    print('\n开始 v26 性能验证...')
    metrics = model.val(data=cfg['data'])

    print(f"\nv26 验证结果:")
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    train_sewer_v26()