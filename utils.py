import base64
import json
import logging
import requests
import tempfile
import os
import re
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，支持numpy数据类型"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def convert_to_serializable(obj):
    """将对象转换为可JSON序列化的类型"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (datetime, Path)):
        return str(obj)
    else:
        return obj


def save_frame_with_metadata(frame, save_dir, filename, metadata=None):
    """
    保存帧图像并可选添加元数据

    Args:
        frame: 图像帧
        save_dir: 保存目录
        filename: 文件名
        metadata: 元数据字典

    Returns:
        str: 保存的文件路径
    """
    save_path = Path(save_dir) / filename

    # 确保目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存图像
    cv2.imwrite(str(save_path), frame)

    # 如果存在元数据，保存为JSON
    if metadata:
        metadata_path = save_path.with_suffix('.json')
        try:
            # 转换元数据为可序列化格式
            metadata_serializable = convert_to_serializable(metadata)

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_serializable, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        except Exception as e:
            logger.warning(f"保存元数据失败: {e}")
            # 尝试简单序列化
            try:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(str(metadata), f, ensure_ascii=False)
            except Exception as e2:
                logger.warning(f"简化保存元数据也失败: {e2}")

    return str(save_path)


def aliyun_ocr_api(image_path, config):
    """
    调用阿里云OCR API

    Args:
        image_path: 图片文件路径
        config: OCR配置字典

    Returns:
        str: OCR识别结果文本
    """
    # 检查配置
    if not config.get("app_code") or "YOUR_APP_CODE" in config.get("app_code", ""):
        logger.warning("阿里云OCR未配置或使用默认值，跳过识别")
        return ""

    # 检查文件是否存在
    if not os.path.exists(image_path):
        logger.error(f"图片文件不存在: {image_path}")
        return ""

    try:
        # 读取图片并编码
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        # 准备请求头
        headers = {
            "Authorization": f"APPCODE {config['app_code']}",
            "Content-Type": "application/json; charset=UTF-8"
        }

        # 准备请求体
        payload = {
            "img": img_b64,
            "prob": True,
            "charInfo": False,
            "rotate": True,
            "table": False
        }

        # 发送请求
        response = requests.post(
            config["url"],
            headers=headers,
            data=json.dumps(payload),
            timeout=config.get("timeout", 10)
        )
        response.raise_for_status()

        # 解析响应
        result = response.json()

        # 不同格式的兼容处理
        if result.get("success"):
            # 格式1: 包含data.content
            lines = result.get("data", {}).get("content", [])
            if isinstance(lines, list):
                return "\n".join(lines)

        # 格式2: 包含words
        words = result.get("words", []) or result.get("prism_wordsInfo", [])
        if isinstance(words, list):
            text_parts = []
            for item in words:
                if isinstance(item, dict):
                    text_parts.append(item.get("word", ""))
            return "\n".join(text_parts)

        # 格式3: 直接返回文本
        if "text" in result:
            return result["text"]

        logger.warning(f"无法解析OCR响应格式: {result}")
        return ""

    except requests.exceptions.Timeout:
        logger.error("OCR请求超时")
        return ""
    except requests.exceptions.ConnectionError:
        logger.error("OCR连接失败，请检查网络")
        return ""
    except requests.exceptions.HTTPError as e:
        logger.error(f"OCR HTTP错误: {e}")
        if e.response.status_code == 401:
            logger.error("OCR认证失败，请检查APPCODE")
        return ""
    except Exception as e:
        logger.error(f"OCR处理异常: {e}")
        return ""


def get_ai_summary_api(ocr_text, config):
    """
    使用AI服务整理OCR结果

    Args:
        ocr_text: OCR识别文本
        config: AI服务配置

    Returns:
        dict: AI整理后的结构化信息
    """
    if not ocr_text or len(ocr_text.strip()) < 10:
        return {
            "任务名称": "未识别",
            "检测地点": "未识别",
            "井号区间": "未识别",
            "管材": "未识别",
            "管径": "未识别",
            "当前距离": "未识别"
        }

    # 提取关键文本（限制长度）
    short_text = ocr_text[:500]

    # 构建提示词
    prompt = f"""
    任务：从管道检测文字中提取关键信息。
    要求：仅输出一个JSON对象，不要任何其他文字。

    待分析文字：
    {short_text}

    输出JSON模板：
    {{
        "任务名称": "",
        "检测地点": "",
        "井号区间": "",
        "管材": "",
        "管径": "",
        "当前距离": ""
    }}

    注意：如果某个信息无法确定，请留空字符串""。
    """

    payload = {
        "model": config.get("model", "gemma3:4b"),
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        # 发送请求
        response = requests.post(
            config["url"],
            json=payload,
            timeout=config.get("timeout", 120)
        )
        response.raise_for_status()

        # 解析响应
        result = response.json()
        result_text = result.get("response", "")

        # 清理JSON响应
        result_text = result_text.strip()

        # 移除可能的代码块标记
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text[3:]

        if result_text.endswith("```"):
            result_text = result_text[:-3]

        result_text = result_text.strip()

        # 解析JSON
        if result_text.startswith("{") and result_text.endswith("}"):
            try:
                return json.loads(result_text)
            except json.JSONDecodeError as e:
                logger.warning(f"AI返回的JSON解析失败: {e}")

        # 尝试提取JSON
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"AI返回格式无效: {result_text[:100]}...")
        return {"error": "AI返回格式无效", "raw_response": result_text[:200]}

    except requests.exceptions.ConnectionError:
        logger.warning("无法连接到AI服务，请确保Ollama已启动")
        return {"error": "AI服务未连接"}
    except requests.exceptions.Timeout:
        logger.error("AI服务请求超时")
        return {"error": "AI服务请求超时"}
    except Exception as e:
        logger.error(f"AI处理失败: {e}")
        return {"error": str(e)}


def calculate_ocr_score(text, patterns=None):
    """
    计算OCR文本的合法性得分

    Args:
        text: OCR识别文本
        patterns: 自定义模式字典

    Returns:
        float: 得分
    """
    if not text:
        return 0.0

    # 默认模式
    if patterns is None:
        patterns = {
            'valid_chars': re.compile(r'[0-9\.mMDL:()\[\]{}，。]|[a-zA-Z\u4e00-\u9fa5]'),
            'numbers': re.compile(r'\d+\.?\d*'),
            'keywords': ['管道', '井号', '距离', '管径', 'mm', 'm', '检测', '井', '管', '深度']
        }

    # 1. 有效字符比例
    valid_chars = patterns['valid_chars'].findall(text)
    valid_ratio = len(valid_chars) / len(text) if text else 0

    # 2. 数字密度（管道检测中数字很重要）
    numbers = patterns['numbers'].findall(text)
    number_density = len(numbers) / max(len(text.split()), 1)

    # 3. 结构合理性（检查常见关键词）
    keyword_score = sum(1 for kw in patterns['keywords'] if kw in text)

    # 4. 行数合理性（太多行可能是噪音）
    lines = text.strip().split('\n')
    line_score = 1.0 - min(len(lines) / 20, 1.0)  # 超过20行扣分

    # 综合得分
    total_score = (
            valid_ratio * 40 +
            number_density * 30 +
            keyword_score * 20 +
            line_score * 10
    )

    return total_score


def cleanup_temp_files():
    """清理临时文件"""
    temp_dir = tempfile.gettempdir()
    pattern = re.compile(r'^tmp.*\.jpg$')

    deleted_count = 0
    for filename in os.listdir(temp_dir):
        if pattern.match(filename):
            try:
                os.unlink(os.path.join(temp_dir, filename))
                deleted_count += 1
            except:
                pass

    if deleted_count > 0:
        logger.debug(f"清理了 {deleted_count} 个临时文件")