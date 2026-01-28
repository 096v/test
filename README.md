# 管道缺陷智能检测系统

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

一个基于深度学习的智能管道缺陷检测系统，集成了目标检测、OCR识别和AI分析功能，能够自动检测管道视频中的缺陷并生成结构化报告。


## 📋 系统要求

### 硬件要求
- **CPU**: Intel i5 或更高
- **内存**: 8GB 以上
- **存储**: 10GB 可用空间
- **GPU** (可选但推荐): NVIDIA显卡 (CUDA 11.0+)

### 软件要求
- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8 或更高版本
- **Ollama**: 用于本地AI服务

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/yourusername/pipe-defect-detection.git
cd pipe-defect-detection
```

### 2. 安装Python依赖
```bash
pip install ultralytics opencv-python requests pillow
```

### 3. 配置本地AI服务
```bash
# 安装Ollama (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh

# 安装Ollama (Windows)
# 从 https://ollama.com/download 下载安装程序

# 启动Ollama服务
ollama serve

# 在新终端中安装模型
ollama pull gemma3:4b
```

### 4. 初始化配置文件
```bash
python Demo.py --create-config
```

### 5. 编辑配置文件
编辑生成的 `config.json` 文件，填入阿里云OCR凭证：

```json
{
  "aliyun_ocr": {
    "app_code": "your_app_code_here",
    "app_key": "your_app_key_here",
    "app_secret": "your_app_secret_here"
  }
}
```

### 6. 准备模型和视频
```
项目目录/
├── weights/
│   └── best.pt          # YOLO模型文件
└── MyVideo_1.mp4        # 检测视频文件
```

### 7. 运行检测
```bash
# 基本检测（不显示画面）
python defect_detector.py

# 显示实时检测画面
python defect_detector.py --show

# 指定视频文件
python defect_detector.py --video path/to/your/video.mp4
```


## ⚙️ 配置说明

### 命令行参数

```bash
python Demo.py --help
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` | 指定视频文件路径 | `MyVideo_1.mp4` |
| `--model` | 指定YOLO模型路径 | `weights/best.pt` |
| `--config` | 指定配置文件路径 | `config.json` |
| `--show` | 显示实时检测画面 | 不显示 |
| `--create-config` | 创建配置文件模板 | 不创建 |

### 配置文件详解

`config.json` 主要配置项：

```json
{
  "video_path": "MyVideo_1.mp4",
  "model_path": "weights/best.pt",
  "frame_save_dir": "frames",
  "output_report": "defect_report.json",
  
  "max_lost_frames": 30,
  "min_duration_frames": 30,
  
  "yolo_confidence": 0.3,
  "yolo_iou": 0.5,
  
  "aliyun_ocr": {
    "app_code": "",
    "app_key": "",
    "app_secret": "",
    "url": "https://gjbsb.market.alicloudapi.com/ocrservice/advanced"
  },
  
  "ai_service": {
    "url": "http://127.0.0.1:11434/api/generate",
    "model": "gemma3:4b",
    "timeout": 120
  }
}
```

### 环境变量配置

对于生产环境，推荐使用环境变量：

```bash
# Linux/macOS
export ALI_APP_CODE="your_app_code"
export ALI_APP_KEY="your_app_key"
export ALI_APP_SECRET="your_app_secret"
export VIDEO_PATH="path/to/video.mp4"

# Windows
set ALI_APP_CODE=your_app_code
set ALI_APP_KEY=your_app_key
set ALI_APP_SECRET=your_app_secret
set VIDEO_PATH=path\to\video.mp4
```

## 📊 输出说明

### 控制台输出示例

```
2024-01-15 10:30:15 - INFO - 开始缺陷检测...
2024-01-15 10:30:20 - INFO - 检测到缺陷: 裂缝, ID:1, 置信度: 0.892, 帧: 100
2024-01-15 10:30:25 - INFO - ✅ ID:1 持续45帧，符合要求，开始调用OCR/AI...
2024-01-15 10:30:35 - INFO -    OCR/AI处理完成
2024-01-15 10:30:35 - INFO -    AI提取信息:
2024-01-15 10:30:35 - INFO -      任务名称: 某市排水管道检测
2024-01-15 10:30:35 - INFO -      检测地点: 中山路
2024-01-15 10:30:35 - INFO -      井号区间: W01-W02
```

### 最终报告总结

```
====================================================================
缺陷检测总结报告
====================================================================
视频文件: MyVideo_1.mp4
总帧数: 1500
持续时间: 50.00秒
帧率: 30.0 FPS

检测到缺陷总数: 8个
详细分析缺陷: 5个
跳过分析缺陷: 3个

缺陷类型统计:
  裂缝: 3个
  渗漏: 2个
  腐蚀: 3个
```

### JSON报告结构

```json
{
  "video_info": {
    "path": "MyVideo_1.mp4",
    "fps": 30.0,
    "total_frames": 1500,
    "total_duration": 50.0
  },
  "detection_config": {
    "model": "weights/best.pt",
    "confidence_threshold": 0.3,
    "iou_threshold": 0.5,
    "max_lost_frames": 30,
    "min_duration_frames": 30
  },
  "defects": [
    {
      "track_id": 1,
      "defect_type": "裂缝",
      "start_frame": 100,
      "end_frame": 145,
      "start_time": 3.33,
      "end_time": 4.83,
      "max_conf": 0.892,
      "ocr_text": "[start_frame_path] 某市排水管道检测...",
      "ai_summary": {
        "任务名称": "某市排水管道检测",
        "检测地点": "中山路",
        "井号区间": "W01-W02",
        "管材": "钢筋混凝土",
        "管径": "DN800",
        "当前距离": "125.3m"
      }
    }
  ],
  "summary": {
    "total_defects": 8,
    "analyzed_defects": 5,
    "defect_types": {
      "裂缝": 3,
      "渗漏": 2,
      "腐蚀": 3
    }
  }
}
```

## 🔧 高级配置

### 调整检测灵敏度

```json
{
  "min_duration_frames": 50,      // 提高阈值减少误报
  "max_lost_frames": 20,          // 更快结算缺陷
  "yolo_confidence": 0.5,         // 提高置信度要求
  "yolo_iou": 0.3                // 减少重叠检测
}
```

### 更换AI模型

```json
{
  "ai_service": {
    "model": "qwen2.5:3b",        // 更换为其他模型
    "timeout": 180                // 增加超时时间
  }
}
```

### 批量处理脚本

创建 `batch_process.py`：

```python
import subprocess
import glob
import json
from datetime import datetime

def batch_process_videos(video_folder, output_folder):
    videos = glob.glob(f"{video_folder}/*.mp4")
    
    for video in videos:
        video_name = video.split('/')[-1].split('.')[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_folder}/report_{video_name}_{timestamp}.json"
        
        cmd = [
            "python", "Demo.py",
            "--video", video,
            "--output", output_file
        ]
        
        print(f"处理视频: {video}")
        subprocess.run(cmd)
        print(f"报告已保存: {output_file}\n")

if __name__ == "__main__":
    batch_process_videos("videos", "reports")
```

## 🐛 故障排除

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|----------|
| 阿里云OCR失败 | 1. APPCODE错误<br>2. 服务欠费<br>3. 网络问题 | 1. 检查配置<br>2. 查看阿里云余额<br>3. 测试网络连接 |
| AI服务连接失败 | 1. Ollama未启动<br>2. 模型未下载 | 1. 运行 `ollama serve`<br>2. 运行 `ollama pull gemma3:4b` |
| 视频无法打开 | 1. 文件路径错误<br>2. 格式不支持<br>3. 解码器缺失 | 1. 检查路径<br>2. 转换为MP4格式<br>3. 安装 `opencv-contrib-python` |
| 检测速度慢 | 1. 视频分辨率过高<br>2. 没有GPU加速 | 1. 降低分辨率<br>2. 安装CUDA版本的PyTorch |

### 日志分析

```bash
# 查看实时日志
tail -f defect_detection.log

# 搜索错误信息
grep -i "error\|fail\|exception" defect_detection.log

# 查看处理进度
grep "已处理" defect_detection.log
```

### 测试各个模块

```python
# 测试OCR模块
python -c "
from Demo import aliyun_ocr
result = aliyun_ocr('test_image.jpg')
print('OCR结果:', result[:100] if result else '空结果')
"

# 测试AI服务
python -c "
from Demo import get_ai_summary
result = get_ai_summary('测试文本：某市管道检测')
print('AI分析结果:', result)
"

# 测试视频读取
python -c "
import cv2
cap = cv2.VideoCapture('MyVideo_1.mp4')
if cap.isOpened():
    print('视频打开成功')
    ret, frame = cap.read()
    print(f'帧读取: {ret}, 尺寸: {frame.shape if ret else "无帧"}')
else:
    print('视频打开失败')
"
```

## 📈 性能优化

### GPU加速

```bash
# 安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证GPU可用性
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 调整处理参数

```json
{
  "frame_save_dir": "/tmp/frames",  // 使用更快存储
  "yolo_confidence": 0.5,           // 减少低置信度处理
  "ai_service": {
    "timeout": 60                   // 减少AI超时时间
  }
}
```

### 内存优化

```python
# 在代码中添加定期清理
import gc

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 添加新的缺陷类型

1. 重新训练YOLO模型，增加新的缺陷类别
2. 更新 `weights/best.pt` 文件
3. 修改配置文件中的类别名称映射

### 集成其他OCR服务

创建新的OCR适配器类：

```python
class BaiduOCR:
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
    
    def recognize(self, image_path):
        # 实现百度OCR接口调用
        pass
```

**type**: feat, fix, docs, style, refactor, test, chore
## 🎯 路线图

- [ ] 训练多个模型
- [x] OCR集成,对缺陷图片进行OCR识别
- [x] AI分析集成


## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [阿里云OCR](https://www.aliyun.com/product/ocr)
- [Ollama](https://ollama.com/)
- [Google Gemma](https://ai.google.dev/gemma)

---

**提示**: 首次使用时，请确保完成以下步骤：
1. 安装Python依赖
2. 配置阿里云OCR凭证
3. 启动Ollama服务并下载模型
4. 准备好视频和模型文件

祝您使用愉快！ 🚀