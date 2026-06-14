
# 问卷星拟人化自动答题工具 🤖

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-lightgrey.svg)]()

基于 **Selenium + PyAutoGUI** 开发的问卷星（wjx.cn）自动化问卷填写系统。项目采用数据驱动设计，引入了高级缓动算法与反指纹技术，能有效应对现代前端风控与滑块验证。

---

## ✨ 核心特性

- **⚙️ YAML 配置驱动** — 问卷链接、题型、逻辑规则全部抽离至 `survey_config.yaml`，**换问卷无需修改任何 Python 代码**。
- **👁️ 高级反检测机制** — 动态伪造 User-Agent 池、底层 CDP 注入抹除 `navigator.webdriver` 标记、随机窗口视口、多层级随机时间扰动。
- **🎢 拟人化滑块拖拽** — 基于 `easeOutCubic` 缓动曲线实现“先快后慢”的非匀速移动，揉入高斯随机垂直抖动与终点微小回退修正，完美模拟人手操作。
- **🧩 复杂题型全覆盖** — 支持 单选 / 多选（数量可控） / 下拉框（JS注入+事件触发） / 矩阵题 / 排序题 / 填空题（动态词库）。
- **🛡️ 鲁棒性保底设计** — 核心交互节点均具备 `Try-Catch` 异常捕获，若高级选择器或 JS 注入失败，将自动降级为物理坐标模拟模式。

---

## 📁 项目结构

```text
my-wjx-bot/
│
├── .gitignore              # Git 忽略规则（防止缓存与本地配置上传）
├── README.md               # 项目说明文档
├── requirements.txt        # Python 依赖包清单
├── survey_config.yaml      # 问卷配置文件（核心：所有题目逻辑在此配置）
└── main.py                 # 自动化脚本主程序入口

```

---

## 🔧 环境要求与安装

### 1. 前置条件

* **Python** ≥ 3.8
* **Microsoft Edge** 或 **Google Chrome** 浏览器（推荐使用 Edge，Windows 系统无需额外配置 WebDriver 路径）

### 2. 快速安装依赖

```bash
pip install -r requirements.txt

```

> **依赖包速查说明：**
> * `selenium`: 负责页面结构解析、隐藏属性抹除及影子点击
> * `pyautogui`: 突破浏览器沙箱，实现操作系统层面的鼠标轨迹物理模拟
> * `pytweening`: 提供符合人类生理习惯的运动学缓动函数（Easing Functions）
> * `pyyaml`: 高效解析轻量级结构化配置文件
> 
> 

---

## 🚀 快速开始

### Step 1：配置全局信息与题目规则

打开 `survey_config.yaml`，按需修改。以下为标准模板：

```yaml
# 全局控制流
settings:
  url: "[https://v.wjx.cn/vm/PyKgkYl.aspx#](https://v.wjx.cn/vm/PyKgkYl.aspx#)"  # 你的目标问卷链接
  submit_selector: "#ctlNext"              # 提交/下一步 按钮的选择器
  success_keyword: "您的答卷已经提交"        # 提交成功页面的校验关键字

# 题目流水线（脚本将严格按照此列表顺序或ID执行操作）
questions:
  - id: "div1"
    type: "dropdown"        # 下拉框题
    desc: "第1题：所属行业"

  - id: "div2"
    type: "radio"           # 单选题
    desc: "第2题：性别"

  - id: "div3"
    type: "checkbox"        # 多选题
    max_choices: 2          # 【可选参数】限制多选时最多随机勾选几项
    desc: "第3题：常用消费渠道"

  - id: "div8"
    type: "matrix"          # 矩阵题
    desc: "第8题：综合满意度矩阵"

  - id: "div12"
    type: "sort"            # 排序题
    desc: "第12题：功能偏好排序"

  - id: "div18"
    type: "text"            # 填空题
    input_id: "q18"         # 映射到真实的 Input 标签 ID
    desc: "第18题：开放性建议"

```

### Step 2：设定运行次数并启动

打开 `main.py`，在底部的程序入口指定你期望提交的次数：

```python
if __name__ == "__main__":
    # 参数 1：循环提交次数；参数 2：配置文件路径
    zonghe(times=10, config_file="survey_config.yaml") 

```

在终端中执行主脚本：

```bash
python main.py

```

---

## 📋 题型配置与行为速查表

| 配置 `type` | 对应题型 | 特有参数 | 内部执行逻辑行为 |
| --- | --- | --- | --- |
| `radio` | **单选题** | — | 自动过滤含有“其他”的干扰项，随机点击 1 个有效选项。 |
| `checkbox` | **多选题** | `max_choices` | 随机勾选 `1` 到 `max_choices` 个选项（防止盲目全选触发风控）。 |
| `dropdown` | **下拉框** | — | 优先通过高级 JS 注入强制修改原生 `select` 索引并触发 `change` 联动事件。 |
| `matrix` | **矩阵题** | — | 按行遍历（`tr` 标签），在每行包含的单选矩阵点中随机勾选。 |
| `sort` | **排序题** | — | 采用 `random.shuffle` 算法在内存中打乱选项顺序后，依次全选点击。 |
| `text` | **填空题** | `input_id` | 90% 概率从动态答案词库（内置50+高频词）中随机抽取填入，10% 概率留空。 |

---

## 🛡️ 反检测防御矩阵

为了绕过问卷星的高强度行为审计与反爬虫引擎，系统在多个层面上进行了“拟人化”加固：

```
[网络请求层] ──> 随机轮替 User-Agent 浏览器指纹检测
      │
[浏览器底层] ──> CDP 动态注入，重写并抹除 navigator.webdriver 痕迹
      │
[视觉框架层] ──> 随机调整初始化视口分辨率 (1000~1400 px)
      │
[行为节奏层] ──> 页面引入随机延迟 (2~5s)、题目切换 (0.5~1.2s)、多轮提交冷却 (10~30s)
      │
[物理操作层] ──> 滑块验证应用 easeOutCubic 曲线 + 高斯垂直微小抖动 + 终点回退对齐

```

---

## 🔍 调试与常见问题 (FAQ)

**Q：滑块验证时鼠标点歪了，或者拖拽距离不够怎么办？**
A：滑块中心点的绝对屏幕坐标受浏览器系统缩放（建议设为 100%）、书签栏高度影响。可以在 `get_element_screen_pos` 函数中微调 `+ 80` （标题栏估算高度）的值，或在 YAML 的降级机制中调整物理点击的边界像素。

**Q：提示“未检测到标准题目元素”或找不到题目？**
A：部分问卷星特殊的主题模板可能会改变类名。建议使用项目配套的元素扫描代码，优先观察控制台输出的 HTML 结构，随后在 `handle_question` 的 `css_selectors` 列表中补充对应的特殊类名。

---

## ⚠️ 免责声明

本公开项目及代码仅用于 **学术研究、Selenium 自动化框架教学 以及 拟人化行为学工程实验**。

* 请勿将本工具用于任何商业刷票、恶意操纵网络问卷数据或扰乱平台正常运营的非法行为。
* 由此产生的任何数据失真、IP 封禁、法律纠纷或民事责任，均由使用者本人承担，项目作者不对此引发的任何后果负任何直接或间接责任。

```

```