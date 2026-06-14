
# 问卷星自动答题脚本

基于 Selenium + PyAutoGUI 的问卷星（wjx.cn）自动化问卷填写工具。

## 功能特性

- **YAML 配置驱动**：题目类型、选项数量等通过 `survey_config.yaml` 配置，无需修改代码
- **反检测机制**：随机 User-Agent、隐藏 WebDriver 属性、随机窗口大小、随机操作延迟
- **人类化轨迹**：滑块验证使用 easeOutCubic 缓动曲线 + 随机抖动，模拟真实拖拽
- **多题型支持**：单选、多选、下拉框、矩阵题、排序题、填空题

## 项目结构

```
├── .gitignore              # Git 忽略文件
├── README.md               # 项目说明书
├── requirements.txt        # 依赖库列表
├── survey_config.yaml      # 问卷配置文件
└── main.py                 # 主程序入口
```

## 环境要求

- Python 3.8+
- Microsoft Edge 浏览器（Windows 自带）

## 安装

```bash
pip install -r requirements.txt
```

## 使用

1. 编辑 `survey_config.yaml`，配置问卷 URL 和题目列表
2. 运行脚本：

```bash
python main.py
```

## 配置说明

`survey_config.yaml` 中每道题支持的 type：

| type | 说明 | 可选参数 |
|------|------|----------|
| `radio` | 单选题 | — |
| `checkbox` | 多选题 | `max_choices`：最多选几项 |
| `dropdown` | 下拉框 | — |
| `matrix` | 矩阵题 | — |
| `sort` | 排序题 | — |
| `text` | 填空题 | `input_id`：输入框 ID（如 `q18`） |

## 免责声明

本工具仅供学习交流使用，请勿用于非法批量刷问卷等行为。
