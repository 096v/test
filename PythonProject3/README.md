# 问卷星自动答题脚本 🤖

基于 **Selenium + PyAutoGUI** 的问卷星（wjx.cn）自动化问卷填写工具，支持多种题型、滑块验证和反检测机制。

## ✨ 功能特性

- **YAML 配置驱动** — 题目类型、选项数量等全部通过 `survey_config.yaml` 配置，换问卷不改代码
- **反检测机制** — 随机 User-Agent 池、隐藏 `navigator.webdriver`、随机窗口大小、随机操作间隔
- **人类化滑块验证** — easeOutCubic 缓动曲线 + 高斯随机抖动 + 回退修正，模拟真实拖拽
- **多题型全覆盖** — 单选 / 多选 / 下拉框 / 矩阵题 / 排序题 / 填空题
- **保底降级方案** — JS 注入失败时自动切换物理点击模式

## 📁 项目结构

```
├── .gitignore              # Git 忽略规则
├── README.md               # 项目说明
├── requirements.txt        # Python 依赖清单
├── survey_config.yaml      # 问卷配置文件（核心：改它即可适配不同问卷）
└── main.py                 # 主程序入口
```

## 🔧 环境要求

- **Python** ≥ 3.8
- **Microsoft Edge** 浏览器（Windows 10/11 自带，无需额外安装）

## 📦 安装

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含：
| 包 | 用途 |
|---|---|
| `selenium` | 浏览器自动化驱动 |
| `pyautogui` | 鼠标键盘模拟（滑块拖拽） |
| `pytweening` | 缓动函数（人类化轨迹） |
| `pyyaml` | YAML 配置文件解析 |

## 🚀 快速开始

### 1. 编辑问卷配置

打开 `survey_config.yaml`，修改 `settings.url` 为目标问卷链接：

```yaml
settings:
  url: "https://v.wjx.cn/vm/你的问卷ID.aspx#"
  submit_selector: "#ctlNext"          # 提交按钮的 CSS 选择器
  success_keyword: "您的答卷已经提交"    # 提交成功页面包含的关键词
```

### 2. 按问卷实际结构配置题目

```yaml
questions:
  - id: "div1"
    type: "dropdown"        # 下拉框
    desc: "第1题：请选择城市"

  - id: "div2"
    type: "radio"           # 单选题
    desc: "第2题：性别"

  - id: "div3"
    type: "checkbox"        # 多选题
    max_choices: 3          # 最多选 3 项
    desc: "第3题：兴趣爱好（多选）"

  - id: "div8"
    type: "matrix"          # 矩阵题
    desc: "第8题：满意度矩阵"

  - id: "div12"
    type: "sort"            # 排序题
    desc: "第12题：按重要性排序"

  - id: "div18"
    type: "text"            # 填空题
    input_id: "q18"         # 输入框的实际 ID
    desc: "第18题：意见建议"
```

### 3. 运行

```bash
python main.py
```

默认执行 1 次。如需批量提交，修改 `main.py` 底部的调用参数：

```python
if __name__ == "__main__":
    zonghe(10, "survey_config.yaml")  # 提交 10 次
```

## 📋 题型配置速查表

| type | 说明 | 特有参数 | 备注 |
|------|------|----------|------|
| `radio` | 单选题 | — | 随机选 1 个可见选项，自动跳过"其他" |
| `checkbox` | 多选题 | `max_choices` | 随机选 1~max_choices 个选项 |
| `dropdown` | 下拉框 | — | JS 注入修改 `<select>`，失败自动降级为物理点击 |
| `matrix` | 矩阵题 | — | 每行随机选一个选项 |
| `sort` | 排序题 | — | 随机打乱顺序后依次点击 |
| `text` | 填空题 | `input_id` | 90% 概率随机填写，10% 留空 |

## 🛡️ 反检测细节

脚本内置多层反自动化检测措施：

| 层级 | 措施 |
|------|------|
| 浏览器指纹 | 随机切换 4 个 User-Agent |
| WebDriver 标记 | CDP 注入脚本隐藏 `navigator.webdriver` |
| 视口特征 | 每次随机 1000-1400 × 800-1000 窗口大小 |
| 时间节奏 | 启动延迟 2-5s，题间间隔 0.5-1.2s，提交间隔 10-30s |
| 操作轨迹 | 滑块使用缓动曲线 + 高斯抖动，非匀速机械拖拽 |

## ⚠️ 免责声明

本工具仅供 **学习交流 Selenium 自动化技术** 使用。请勿用于非法批量刷问卷、破坏数据真实性等行为。使用者需自行承担一切法律责任。
