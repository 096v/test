# utils.py
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
from paddleocr import PaddleOCR

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


class PaddleOCRWrapper:
    """PaddleOCR包装类"""

    def __init__(self, config=None):
        """
        初始化PaddleOCR

        Args:
            config: PaddleOCR配置
        """
        self.config = config or {}

        # 设置环境变量解决可能的冲突
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # PaddleOCR配置参数
        use_angle_cls = self.config.get("use_angle_cls", True)
        lang = self.config.get("lang", "ch")
        use_gpu = self.config.get("use_gpu", False)
        gpu_mem = self.config.get("gpu_mem", 500)
        cpu_threads = self.config.get("cpu_threads", 10)
        det_db_thresh = self.config.get("det_db_thresh", 0.3)
        det_db_box_thresh = self.config.get("det_db_box_thresh", 0.6)
        drop_score = self.config.get("drop_score", 0.5)
        show_log = self.config.get("show_log", False)

        logger.info(f"初始化PaddleOCR: lang={lang}, use_gpu={use_gpu}, use_angle_cls={use_angle_cls}")

        try:
            self.ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                use_gpu=use_gpu,
                gpu_mem=gpu_mem,
                cpu_threads=cpu_threads,
                det_db_thresh=det_db_thresh,
                det_db_box_thresh=det_db_box_thresh,
                drop_score=drop_score,
                show_log=show_log
            )
            logger.info("PaddleOCR初始化成功")
        except Exception as e:
            logger.error(f"PaddleOCR初始化失败: {e}")
            # 尝试简化配置初始化
            try:
                self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=show_log)
                logger.info("PaddleOCR使用简化配置初始化成功")
            except Exception as e2:
                logger.error(f"PaddleOCR简化初始化也失败: {e2}")
                raise

    def ocr_image(self, image_path):
        """
        对图片进行OCR识别

        Args:
            image_path: 图片路径或numpy数组

        Returns:
            str: 识别出的文本
        """
        try:
            # 如果传入的是numpy数组，直接使用
            if isinstance(image_path, np.ndarray):
                result = self.ocr.ocr(image_path, cls=True)
            else:
                # 检查文件是否存在
                if not os.path.exists(image_path):
                    logger.error(f"图片文件不存在: {image_path}")
                    return ""
                result = self.ocr.ocr(image_path, cls=True)

            if not result or not result[0]:
                return ""

            # 提取所有文本
            texts = []
            for line in result[0]:
                if line and len(line) > 1:
                    text = line[1][0]
                    confidence = line[1][1] if len(line[1]) > 1 else 0.0
                    # 根据置信度筛选
                    if confidence >= self.config.get("min_confidence", 0.5):
                        texts.append(text)
                    else:
                        logger.debug(f"低置信度文本跳过: {text} (置信度: {confidence:.3f})")

            # 合并文本
            ocr_result = "\n".join(texts)
            logger.debug(f"OCR识别结果: {ocr_result[:100]}...")
            return ocr_result

        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            return ""

    def ocr_frame(self, frame):
        """
        对视频帧进行OCR识别（直接传入numpy数组）

        Args:
            frame: OpenCV图像帧（numpy数组）

        Returns:
            str: 识别出的文本
        """
        try:
            result = self.ocr.ocr(frame, cls=True)

            if not result or not result[0]:
                return ""

            # 提取所有文本
            texts = []
            for line in result[0]:
                if line and len(line) > 1:
                    text = line[1][0]
                    confidence = line[1][1] if len(line[1]) > 1 else 0.0
                    # 根据置信度筛选
                    if confidence >= self.config.get("min_confidence", 0.5):
                        texts.append(text)

            # 合并文本
            ocr_result = "\n".join(texts)
            return ocr_result

        except Exception as e:
            logger.error(f"帧OCR识别失败: {e}")
            return ""


# 全局OCR实例
_paddle_ocr_instance = None


def get_paddle_ocr_instance(config=None):
    """
    获取PaddleOCR实例（单例模式）

    Args:
        config: OCR配置

    Returns:
        PaddleOCRWrapper实例
    """
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        _paddle_ocr_instance = PaddleOCRWrapper(config)
    return _paddle_ocr_instance


def paddle_ocr_api(image_path, config=None):
    """
    PaddleOCR API接口（兼容原有接口）

    Args:
        image_path: 图片文件路径或numpy数组
        config: OCR配置字典

    Returns:
        str: OCR识别结果文本
    """
    try:
        ocr_instance = get_paddle_ocr_instance(config)
        return ocr_instance.ocr_image(image_path)
    except Exception as e:
        logger.error(f"PaddleOCR处理异常: {e}")
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