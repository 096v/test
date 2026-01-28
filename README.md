# 🚀 管道缺陷智能检测调试版

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-v10-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**基于深度学习的自动化管道检测解决方案**

[项目演示](#-演示) | [安装指南](#-安装指南) | [使用说明](#-使用说明) | [配置说明](#-配置说明) | [故障排除](#-故障排除)

</div>

## 📋 目录

- [项目概述](#-项目概述)
- [核心功能](#-核心功能)
- [系统架构](#-系统架构)
- [安装指南](#-安装指南)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [使用方法](#-使用方法)
- [检测报告](#-检测报告)
- [故障排除](#-故障排除)
- [开发说明](#-开发说明)
- [许可证](#-许可证)

## 🎯 项目概述

**管道缺陷智能检测系统**是一个基于YOLOv8深度学习模型的自动化管道健康检测解决方案。该系统能够自动识别视频中的管道缺陷，集成镜像纠正、文字识别和智能分析功能，为管道维护提供全面的技术支持。

### 应用场景

- 🏢 **市政管道检测** - 城市排水、供水管道健康状况评估
- 🏭 **工业管道监测** - 工厂管道系统定期维护检查
- 🚇 **地下管网巡查** - 地铁、隧道等地下管道安全检测
- 🔧 **设施维护** - 建筑物内部管道系统预防性维护

## ✨ 核心功能

| 功能模块 | 技术实现 | 主要特点 |
|---------|---------|---------|
| **缺陷检测** | YOLOv10 + ByteTrack | 15种缺陷类型识别，实时目标追踪 |
| **镜像纠正** | 边缘能量分析 + OCR验证 | 自动检测并纠正视频镜像翻转 |
| **文字识别** | 阿里云OCR API | 提取管道检测中的关键信息 |
| **智能分析** | Ollama AI服务 | 结构化提取检测参数 |
| **报告生成** | JSON格式导出 | 完整的检测过程记录和统计 |
| **可视化** | OpenCV实时显示 | 实时标注和状态监控 |

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      应用层 (Application)                    │
├─────────────────────────────────────────────────────────────┤
│ 命令行接口 │ 视频处理 │ 报告生成 │ 可视化界面 │ 数据导出        │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                     服务层 (Services)                        │
├─────────────────────────────────────────────────────────────┤
│ 缺陷检测服务 │ 镜像纠正服务 │ OCR服务 │ AI分析服务 │ 文件服务    │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                     算法层 (Algorithms)                      │
├─────────────────────────────────────────────────────────────┤
│ YOLOv8检测 │ ByteTrack追踪 │ 能量分析 │ 模式识别 │ 图像处理    │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                     数据层 (Data)                           │
├─────────────────────────────────────────────────────────────┤
│ 视频流 │ 图像帧 │ 模型权重 │ 配置文件 │ 检测结果 │ 日志文件    │
└─────────────────────────────────────────────────────────────┘
```

## 📦 安装指南

### 系统要求

- **操作系统**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 或更高版本
- **内存**: 最低4GB RAM (推荐8GB)
- **存储**: 至少2GB可用空间
- **GPU**: 可选，CUDA 11.0+支持GPU加速

### 环境搭建

#### 1. 克隆项目

```bash
git clone https://github.com/096v/test.git
```

#### 2. 创建虚拟环境（推荐）

**Windows (Anaconda):**
```bash
conda create -n pipe_yolo_v2 python=3.9
conda activate pipe_yolo_v2
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 模型准备

1. **下载预训练模型**: 将YOLO模型文件(`best.pt`)放置在`weights/`目录下
2. **模型来源**: 
   - 使用已训练的管道缺陷检测模型
   - 或根据[训练指南](#模型训练)训练自定义模型

### 可选服务配置

#### OCR服务（阿里云）
1. 访问[阿里云市场](https://market.aliyun.com/products/57124001/cmapi029925.html)
2. 购买"通用文字识别"服务
3. 在`config.json`中配置AppCode

#### AI服务（Ollama）
1. 下载并安装[Ollama](https://ollama.ai/download)
2. 启动Ollama服务并安装模型：
```bash
ollama pull gemma:2b
# 或
ollama pull llama2:7b
```

## 🚀 快速开始

### 第一步：初始化配置

```bash
# 生成配置文件模板
python main.py --create-config
```

编辑生成的`config.json`文件：
```json
{
  "video_path": "your_video.mp4",
  "model_path": "weights/best.pt",
  "frame_save_dir": "frames",
  "output_report": "defect_report.json",
  "yolo_confidence": 0.3,
  "mirror_detection": {
    "enabled": true,
    "calibration_seconds": 5
  }
}
```

### 第二步：运行检测

```bash
# 基本检测（无显示窗口）
python main.py --video Video_1.mp4

# 显示实时检测窗口
python main.py --video Video_1.mp4 --show

# 禁用镜像检测功能
python main.py --video Video_1.mp4 --no-mirror

# 使用详细日志输出
python main.py --video Video_1.mp4 --verbose
```

### 第三步：查看结果

检测完成后：
1. 查看日志文件：`defect_detection.log`
2. 查看检测报告：`defect_report.json`
3. 查看保存的图像帧：`frames/`目录

## ⚙️ 配置说明

### 配置文件详解

`config.json`包含以下主要配置节：

#### 1. 基础配置
```json
{
  "video_path": "Video_1.mp4",
  "model_path": "weights/best.pt",
  "frame_save_dir": "frames",
  "output_report": "defect_report.json",
  "max_lost_frames": 30,
  "min_duration_frames": 30,
  "yolo_confidence": 0.3,
  "yolo_iou": 0.5
}
```

#### 2. 镜像检测配置
```json
"mirror_detection": {
  "enabled": true,
  "calibration_seconds": 5,
  "energy_deviation_threshold": 3.0,
  "ocr_check_interval": 30
}
```

#### 3. 服务配置
```json
"aliyun_ocr": {
  "app_code": "your_app_code_here",
  "url": "https://gjbsb.market.alicloudapi.com/ocrservice/advanced"
},
"ai_service": {
  "url": "http://127.0.0.1:11434/api/generate",
  "model": "gemma3:4b",
  "timeout": 120
}
```

### 环境变量配置

支持使用环境变量覆盖配置：

```bash
# 设置OCR服务
export ALI_APP_CODE="your_app_code"
export ALI_APP_KEY="your_app_key"
export ALI_APP_SECRET="your_app_secret"

# 设置AI服务
export AI_SERVICE_URL="http://127.0.0.1:11434/api/generate"
export AI_MODEL="gemma3:4b"

# 设置镜像检测
export MIRROR_DETECTION_ENABLED="true"
export MIRROR_CALIBRATION_SECONDS="5"

# 设置检测参数
export YOLO_CONFIDENCE="0.3"
export MAX_LOST_FRAMES="30"
```

## 📖 使用方法

### 命令行参数

| 参数 | 缩写 | 描述 | 示例 |
|------|------|------|------|
| `--video` | - | 视频文件路径 | `--video test.mp4` |
| `--model` | - | YOLO模型路径 | `--model custom.pt` |
| `--config` | - | 配置文件路径 | `--config my_config.json` |
| `--show` | - | 显示检测窗口 | `--show` |
| `--create-config` | - | 创建配置文件 | `--create-config` |
| `--no-mirror` | - | 禁用镜像检测 | `--no-mirror` |
| `--verbose` | `-v` | 详细日志输出 | `--verbose` |
| `--ascii-only` | - | 仅ASCII输出 | `--ascii-only` |

### 使用示例

#### 示例1：完整检测流程
```bash
# 创建配置文件（首次使用）
python main.py --create-config

# 编辑config.json，填入您的配置

# 运行完整检测（包含镜像纠正、OCR、AI分析）
python main.py --video inspection.mp4 --show --verbose
```

#### 示例2：批量处理
```bash
# 批量处理多个视频文件
for video in videos/*.mp4; do
    echo "正在处理: $video"
    python main.py --video "$video" --config config.json
done
```

#### 示例3：仅检测（禁用额外服务）
```bash
# 仅进行缺陷检测，不进行OCR和AI分析
python main.py --video inspection.mp4 --no-mirror
```

### 输出文件说明

- **`defect_detection.log`**: 详细的运行日志
- **`defect_report.json`**: 完整的检测报告
- **`frames/`**: 保存的关键帧图像和元数据
- **临时文件**: 系统自动清理OCR验证的临时文件

## 📊 检测报告

### 报告结构

```json
{
  "generated_at": "2026-01-28 21:43:27",
  "video_info": {
    "path": "Video_1.mp4",
    "resolution": "1280x720",
    "fps": 25.0,
    "total_frames": 486,
    "total_duration": 19.44
  },
  "detection_config": {
    "model": "weights/best.pt",
    "confidence_threshold": 0.3,
    "iou_threshold": 0.5
  },
  "mirror_correction": {
    "enabled": true,
    "status": {
      "is_mirrored": false,
      "is_calibrated": true,
      "normal_imbalance_ratio": 1.874
    }
  },
  "defects": [
    {
      "track_id": 3,
      "defect_type": "PL",
      "start_frame": 50,
      "end_frame": 99,
      "start_time": 2.0,
      "end_time": 3.96,
      "max_conf": 0.95,
      "ocr_text": "检测位置: 井号3-4...",
      "ai_summary": {
        "任务名称": "管道检测",
        "检测地点": "XX路段",
        "井号区间": "3-4",
        "管材": "PVC",
        "管径": "300mm",
        "当前距离": "125.3m"
      }
    }
  ],
  "summary": {
    "total_defects": 15,
    "analyzed_defects": 5,
    "defect_types": {
      "PL": 3,
      "SG": 2,
      "CJ": 1
    }
  }
}
```

### 统计信息

系统会自动统计：
- ✅ 检测到的缺陷总数
- 📈 各类型缺陷分布
- ⏱️ 每个缺陷的持续时间和位置
- 🔍 OCR识别的文本信息
- 🧠 AI提取的结构化数据

## 🔧 故障排除

### 常见问题

#### 1. Unicode编码错误（Windows）
**问题**: Windows控制台显示编码错误
**解决方案**:
```bash
# 方法1：使用--ascii-only参数
python main.py --video test.mp4 --ascii-only

# 方法2：配置PowerShell使用UTF-8
$OutputEncoding = [console]::InputEncoding = [console]::OutputEncoding = New-Object System.Text.UTF8Encoding
```

#### 2. AI服务连接失败
**问题**: 无法连接到Ollama服务
**解决方案**:
1. 确保Ollama服务已启动
```bash
ollama serve
```
2. 或禁用AI服务（编辑config.json）：
```json
"ai_service": {
  "url": "",
  "model": "",
  "timeout": 120
}
```

#### 3. 内存不足
**问题**: 处理大型视频时内存不足
**解决方案**:
```bash
# 降低检测精度
yolo_confidence: 0.5  # 增加置信度阈值

# 减少保存的帧图像
min_duration_frames: 50  # 增加最小持续时间
```

#### 4. 视频打开失败
**问题**: 无法打开视频文件
**解决方案**:
1. 检查文件路径是否正确
2. 确保视频格式受支持（MP4, AVI, MOV）
3. 安装必要的编解码器
```bash
pip install opencv-contrib-python
```

### 性能优化

| 场景 | 优化建议 | 预期效果 |
|------|---------|---------|
| **CPU模式** | 降低检测帧率，减少OCR调用 | 减少30-50%CPU使用 |
| **GPU加速** | 启用CUDA，批量处理 | 提升3-5倍处理速度 |
| **内存优化** | 限制保存帧数，使用缓存 | 减少50%内存使用 |
| **网络优化** | 本地化AI服务，减少API调用 | 提升响应速度 |

## 💻 开发说明

### 项目结构

```
pipeline_defect_detection/
├── main.py              # 主程序入口
├── config_loader.py     # 配置管理模块
├── mirror_manager.py    # 镜像检测与纠正
├── utils.py             # 工具函数和API接口
├── requirements.txt     # 依赖包列表
├── config.json         # 配置文件
├── weights/            # 模型目录
├── frames/             # 图像存储
├── tests/              # 测试文件
└── docs/               # 文档
```

### 模块说明

#### `main.py`
- 主控制逻辑
- 缺陷检测器类
- 视频处理循环
- 报告生成

#### `mirror_manager.py`
- 镜像状态检测
- 边缘能量分析
- OCR验证逻辑
- 自动纠正实现

#### `config_loader.py`
- 配置文件读取
- 环境变量处理
- 配置验证
- 默认值管理

#### `utils.py`
- OCR服务接口
- AI服务接口
- 图像处理工具
- 数据序列化

### 扩展开发

#### 添加新的缺陷类型
1. 重新训练YOLO模型，添加新类别
2. 更新`weights/best.pt`模型文件
3. 系统会自动识别新类别

#### 集成新的OCR服务
1. 在`utils.py`中添加新的OCR函数
2. 修改`config.json`添加相应配置
3. 更新`main.py`中的OCR调用

### 模型训练

如需训练自定义模型：

```bash
# 1. 准备数据集
#    - 标注管道缺陷图像
#    - 使用LabelImg或CVAT标注工具

# 2. 训练YOLO模型
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=100 imgsz=640

# 3. 导出模型
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

## 📱 支持的缺陷类型

系统目前支持检测15种管道缺陷：

| 代码 | 中文名称 | 英文名称 | 描述 |
|------|---------|---------|------|
| PL | 破裂 | Crack | 管道壁出现裂缝或断裂 |
| SG | 变形 | Deformation | 管道形状发生改变 |
| CJ | 错口 | Misalignment | 管段连接处不对齐 |
| ZW | 障碍物 | Obstruction | 管道内有异物堵塞 |
| BX | 变形 | Buckling | 管道受压变形 |
| AJ | 腐蚀 | Corrosion | 管道表面腐蚀 |
| QF | 起伏 | Undulation | 管道线路起伏不平 |
| CK | 穿孔 | Perforation | 管道壁有穿孔 |
| TJ | 脱节 | Disconnection | 管道连接处脱开 |
| SL | 渗漏 | Leakage | 管道有渗漏现象 |
| FS | 腐蚀 | Erosion | 管道内壁磨损 |
| TL | 脱落 | Spalling | 管道内衬脱落 |
| CR | 裂纹 | Fissure | 细小裂纹 |
| JG | 结垢 | Scaling | 管道内壁结垢 |
| CQ | 穿孔 | Hole | 较大的穿孔 |

## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 优秀的深度学习框架
- [阿里云OCR](https://www.aliyun.com/product/ocr) - 文字识别服务
- [Ollama](https://ollama.ai/) - 本地AI模型运行环境
- [OpenCV](https://opencv.org/) - 计算机视觉库

## 📞 联系方式

如有问题或建议，请通过以下方式联系我们：

- **GitHub Issues**: [提交问题](https://github.com/096v/Pipeline-DetectionOutlook/issues)
- **电子邮件**: numup@foxmail.com
- **项目主页**: https://github.com/096v/test
---

<div align="center">
  
**感谢使用管道缺陷智能检测系统！**

</div>
