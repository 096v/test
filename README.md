# 🚀 管道缺陷智能检测系统（PaddleOCR版）

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-v8-red)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-本地OCR-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**基于YOLOv8+PaddleOCR的管道缺陷智能检测解决方案**

[快速开始](#-快速开始) | [系统架构](#🏗️-系统架构) | [配置说明](#⚙️-配置说明) | [故障排除](#🔧-故障排除)

</div>

## 🎯 项目概述

**管道缺陷智能检测系统**是基于YOLOv8深度学习模型的自动化管道健康检测解决方案。系统集成本地PaddleOCR，无需网络连接即可实现缺陷检测、文字识别和智能分析。

## ✨ 核心功能

| 功能模块 | 技术实现 | 主要特点 |
|---------|---------|---------|
| **缺陷检测** | YOLOv10 + ByteTrack | 15种缺陷类型识别，实时目标追踪 |
| **镜像纠正** | 边缘能量分析 + OCR验证 | 自动检测并纠正视频镜像翻转 |
| **文字识别** | PaddleOCR本地识别 | 完全离线，无需API密钥 |
| **智能分析** | Ollama AI服务 | 结构化提取检测参数 |
| **报告生成** | JSON格式导出 | 完整的检测过程记录 |

## 🏗️ 系统架构

### 系统职责边界

**Python（算法与感知层）**
- 视频解码与帧级处理
- YOLOv8缺陷检测与追踪（trace_id）
- 镜像异常检测（算法层）
- OCR文字识别（输出原始文本）
- 输出原始、客观的检测事实数据

**Java（业务与平台层）**
- 长视频切分与任务调度
- 缺陷事件合并与连续性判断
- 缺陷类型映射与伴生缺陷规则
- 缺陷密度、修复指数等指标计算
- 多路检测报告生成
- LLM调用与结果兜底

## 📦 安装指南

### 环境搭建

```bash
# 克隆项目
git clone https://github.com/096v/test.git

# 安装依赖
pip install -r requirements.txt
```

**requirements.txt 内容：**
```txt
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.19.0
paddlepaddle>=2.6.0
paddleocr>=2.7.0
requests>=2.25.0
pillow>=8.0.0
```

## 🚀 快速开始

### 第一步：初始化配置

```bash
# 生成配置文件模板
python main.py --create-config
```

### 第二步：运行检测

```bash
# 基本检测
python main.py --video Video_1.mp4

# 显示实时检测窗口
python main.py --video Video_1.mp4 --show
```

### 第三步：查看结果

检测完成后：
- 查看日志文件：`defect_detection.log`
- 查看检测报告：`defect_report.json`
- 查看保存的图像帧：`frames/`目录

## ⚙️ 配置说明

### 基础配置
```json
{
  "video_path": "Video_1.mp4",
  "model_path": "weights/best.pt",
  "paddle_ocr": {
    "lang": "ch",
    "use_gpu": false,
    "min_confidence": 0.5
  }
}
```

### PaddleOCR配置详解

```json
"paddle_ocr": {
  "use_angle_cls": true,      // 启用方向分类器
  "lang": "ch",               // 语言：ch中文, en英文
  "use_gpu": false,           // 使用GPU加速
  "min_confidence": 0.5,      // 置信度阈值
  "show_log": false           // 关闭日志输出
}
```

## 🎞️ 长视频处理策略

### 设计策略
- 由平台侧将原始视频按大小或时长切分为多个片段
- 每个片段作为独立检测任务调用算法服务
- 算法侧仅对单段视频负责，不感知全局视频关系

### 跨段问题说明
- 当前版本不支持跨视频段缺陷自动合并
- 该能力依赖全局里程标定与设备位姿对齐，已纳入后续规划

## 🔗 缺陷连续性建模（trace_id）

### 算法侧输出
- trace_id（目标追踪ID）
- 起始帧/结束帧
- 缺陷类型（label）
- 置信度分数

### 平台侧处理
- 根据trace_id与时间间隔规则合并为单一缺陷事件
- 支持连续缺陷的时长统计与位置计算

## 🧩 缺陷类型与伴生缺陷

### 算法标签（Algorithm Label）
- YOLOv8输出的原始缺陷类别
- 用于模型训练与推理

### 业务缺陷类型（Business Type）
- 由平台侧根据规则进行映射
- 支持多标签组合形成伴生缺陷

> 算法标签与业务缺陷类型解耦设计，便于模型演进与规则调整。

## 📝 OCR与文字语义处理

### OCR识别
- 由Python算法侧完成
- 输出原始文本与置信度
- 支持多语言识别

### 文本理解
- 由平台侧或AI服务完成
- 对非结构化文本进行字段提取与总结

> OCR仅负责感知层识别，语义理解不属于算法职责。

## 🧠 LLM使用说明

本系统中的LLM属于辅助增强能力：
- 用于文本总结、报告生成与说明性输出
- 不参与缺陷检测主流程
- 调用失败不会影响检测结果

> 检测结果的权威来源始终是算法模型与规则计算。

## 🔧 故障排除

### 常见问题

#### 1. 首次运行下载模型慢
**解决方案**：首次运行需下载OCR模型（约200MB），可设置国内镜像源。

#### 2. GPU内存不足
**解决方案**：限制GPU内存使用
```json
"paddle_ocr": {
  "use_gpu": true,
  "gpu_mem": 500
}
```

#### 3. 中文识别效果不佳
**解决方案**：调整识别参数
```json
"paddle_ocr": {
  "det_db_thresh": 0.2,
  "det_db_box_thresh": 0.5,
  "drop_score": 0.3
}
```

## 📞 联系方式

- **GitHub Issues**: [提交问题](https://github.com/096v/Pipeline-DetectionOutlook/issues)
- **电子邮件**: numup@foxmail.com
- **项目主页**: https://github.com/096v/test

---

<div align="center">

**感谢使用管道缺陷智能检测系统！**

</div>
