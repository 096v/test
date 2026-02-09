#utils.py
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
import io
from PIL import Image
import time
from typing import List, Dict, Tuple, Any, Optional

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


def image_to_base64(image, quality=85, max_size=None):
    """
    将图像转换为Base64编码字符串

    Args:
        image: numpy数组或PIL图像
        quality: JPEG质量 (1-100)
        max_size: 最大尺寸 (宽度, 高度)，None表示不调整大小

    Returns:
        Base64编码的字符串
    """
    try:
        # 处理numpy数组
        if isinstance(image, np.ndarray):
            # OpenCV格式 (BGR) 转换为RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # 调整图像大小
            if max_size and (image_rgb.shape[1] > max_size[0] or image_rgb.shape[0] > max_size[1]):
                height, width = image_rgb.shape[:2]
                scale = min(max_size[0] / width, max_size[1] / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))

            # 转换为PIL图像
            pil_image = Image.fromarray(image_rgb)
        elif isinstance(image, Image.Image):
            pil_image = image
            # 调整图像大小
            if max_size:
                pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        else:
            raise ValueError("不支持的图像格式")

        # 将图像保存到内存缓冲区
        buffer = io.BytesIO()

        # 根据图像模式选择格式，默认JPEG
        if pil_image.mode == 'RGBA':
            pil_image.save(buffer, format='PNG')
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            mime_type = 'image/png'
        else:
            # 默认使用JPEG格式（更小）
            pil_image.save(buffer, format='JPEG', quality=quality)
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            mime_type = 'image/jpeg'

        return f"data:{mime_type};base64,{base64_str}"

    except Exception as e:
        logger.error(f"图像转换为Base64失败: {e}")
        return ""


def save_frame_with_metadata(frame, save_dir, filename, metadata=None, generate_base64=False, base64_config=None):
    """
    保存帧图像并可选添加元数据和Base64编码

    Args:
        frame: 图像帧
        save_dir: 保存目录
        filename: 文件名
        metadata: 元数据字典
        generate_base64: 是否生成Base64编码
        base64_config: Base64配置字典

    Returns:
        dict: 包含文件路径和Base64编码的信息字典
    """
    save_path = Path(save_dir) / filename

    # 确保目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存图像
    cv2.imwrite(str(save_path), frame)

    # 准备返回结果
    result = {
        "filepath": str(save_path),
        "filename": filename,
        "width": frame.shape[1],
        "height": frame.shape[0],
        "channels": frame.shape[2] if len(frame.shape) == 3 else 1
    }

    # 生成Base64编码
    if generate_base64 and base64_config is not None:
        quality = base64_config.get("quality", 85)
        max_size = base64_config.get("max_size")
        result["base64"] = image_to_base64(frame, quality=quality, max_size=max_size)

    # 如果存在元数据，保存为JSON
    if metadata:
        metadata_path = save_path.with_suffix('.json')
        try:
            # 转换元数据为可序列化格式
            metadata_serializable = convert_to_serializable(metadata)

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_serializable, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

            result["metadata_path"] = str(metadata_path)
        except Exception as e:
            logger.warning(f"保存元数据失败: {e}")
            # 尝试简单序列化
            try:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(str(metadata), f, ensure_ascii=False)
                result["metadata_path"] = str(metadata_path)
            except Exception as e2:
                logger.warning(f"简化保存元数据也失败: {e2}")

    return result


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
    if not ocr_text or len(ocr_text.strip()) < 5:
        return {
            "startWellNumber": "",
            "endWellNumber": "",
            "pipelineType": "",
            "pipelineMaterial": "",
            "inspectionDirection": "",
            "inspectionDate": "",
            "inspector": ""
        }

    # 提取关键文本（限制长度）
    short_text = ocr_text[:500]

    # 构建提示词 - 要求返回英文字段名
    prompt = f"""
    任务：从管道检测文字中提取以下关键信息。
    要求：严格输出 JSON 对象，字段使用英文，若无法确定则留空字符串 ""。

待分析文字：
{short_text}

输出 JSON 模板：
{{
    "startWellNumber": "", 
    "endWellNumber": "", 
    "pipelineType": "", 
    "pipelineMaterial": "", 
    "inspectionDirection": "", 
    "inspectionDate": "", 
    "inspector": ""
}}

注意：
1. 字段名必须是英文
2. 如果某个信息无法确定，请留空字符串""
3. 检测日期格式应为"YYYY-MM-DD"或留空
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
                ai_result = json.loads(result_text)
                # 确保所有必需字段都存在
                required_fields = ["startWellNumber", "endWellNumber", "pipelineType",
                                   "pipelineMaterial", "inspectionDirection", "inspectionDate", "inspector"]
                for field in required_fields:
                    if field not in ai_result:
                        ai_result[field] = ""
                return ai_result
            except json.JSONDecodeError as e:
                logger.warning(f"AI返回的JSON解析失败: {e}")

        # 尝试提取JSON
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                ai_result = json.loads(json_match.group())
                # 确保所有必需字段都存在
                required_fields = ["startWellNumber", "endWellNumber", "pipelineType",
                                   "pipelineMaterial", "inspectionDirection", "inspectionDate", "inspector"]
                for field in required_fields:
                    if field not in ai_result:
                        ai_result[field] = ""
                return ai_result
            except json.JSONDecodeError:
                pass

        logger.warning(f"AI返回格式无效: {result_text[:100]}...")
        # 返回空字段结构
        return {
            "startWellNumber": "",
            "endWellNumber": "",
            "pipelineType": "",
            "pipelineMaterial": "",
            "inspectionDirection": "",
            "inspectionDate": "",
            "inspector": ""
        }

    except requests.exceptions.ConnectionError:
        logger.warning("无法连接到AI服务，请确保Ollama已启动")
        return {
            "error": "AI服务未连接",
            "startWellNumber": "",
            "endWellNumber": "",
            "pipelineType": "",
            "pipelineMaterial": "",
            "inspectionDirection": "",
            "inspectionDate": "",
            "inspector": ""
        }
    except requests.exceptions.Timeout:
        logger.error("AI服务请求超时")
        return {
            "error": "AI服务请求超时",
            "startWellNumber": "",
            "endWellNumber": "",
            "pipelineType": "",
            "pipelineMaterial": "",
            "inspectionDirection": "",
            "inspectionDate": "",
            "inspector": ""
        }
    except Exception as e:
        logger.error(f"AI处理失败: {e}")
        return {
            "error": str(e),
            "startWellNumber": "",
            "endWellNumber": "",
            "pipelineType": "",
            "pipelineMaterial": "",
            "inspectionDirection": "",
            "inspectionDate": "",
            "inspector": ""
        }


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


def evaluate_ocr_quality(text, patterns=None):
    """
    评估OCR文本质量，返回评分和细节

    Args:
        text: OCR识别文本
        patterns: 自定义模式字典

    Returns:
        dict: 包含评分和细节的字典
    """
    if not text:
        return {"score": 0.0, "details": {}, "keywords_found": []}

    # 默认管道检测关键词
    if patterns is None:
        patterns = {
            'keywords': [
                "井", "雨", "污", "管", "道", "检测", "S", "Q", "Φ", "DN",
                "上游", "下游", "起始", "终止", "编号", "人员", "日期",
                "雨水", "污水", "给水", "管道", "井号", "材质", "方向",
                "检测井", "检查井", "管径", "深度", "长度", "坐标"
            ],
            'date_patterns': [
                r'\d{4}[-年]\d{1,2}[-月]\d{1,2}[日]?',
                r'\d{4}/\d{1,2}/\d{1,2}'
            ],
            'well_patterns': [
                r'[SQ]\d{3,}',
                r'井[号]?[:：]?\s*[A-Z]?\d+',
                r'[起终][始点][井]?[:：]?\s*[A-Z]?\d+'
            ]
        }

    # 1. 文本长度得分
    text_length = len(text)
    max_expected_len = 1000
    length_score = min(text_length / max_expected_len, 1.0)

    # 2. 关键词匹配得分
    keyword_hits = sum(1 for kw in patterns['keywords'] if kw in text)
    keyword_score = min(keyword_hits / len(patterns['keywords']) * 2, 1.0)  # *2因为有些关键词可能同时出现

    # 3. 数字密度得分（管道检测中数字很重要）
    digit_count = sum(c.isdigit() for c in text)
    digit_score = min(digit_count / max(text_length, 1) * 10, 1.0)

    # 4. 结构合理性（行数、标点等）
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    line_count = len(lines)

    # 检查是否有井号、日期等关键信息
    has_well_number = any(re.search(pattern, text) for pattern in patterns['well_patterns'])
    has_date = any(re.search(pattern, text) for pattern in patterns['date_patterns'])

    structure_score = 0.0
    if 3 <= line_count <= 15:  # 合理的行数范围
        structure_score += 0.3
    if has_well_number:
        structure_score += 0.4
    if has_date:
        structure_score += 0.3

    # 5. 有效字符比例
    valid_chars = re.findall(r'[0-9a-zA-Z\u4e00-\u9fa5，。、；：！？（）《》【】·\-\./:]', text)
    valid_ratio = len(valid_chars) / max(text_length, 1)

    # 计算加权总分
    weights = {
        'length': 0.15,
        'keyword': 0.45,  # 关键词权重最高
        'digit': 0.20,
        'structure': 0.15,
        'valid_ratio': 0.05
    }

    total_score = (
            length_score * weights['length'] +
            keyword_score * weights['keyword'] +
            digit_score * weights['digit'] +
            structure_score * weights['structure'] +
            valid_ratio * weights['valid_ratio']
    )

    return {
        "score": total_score,
        "details": {
            "text_length": text_length,
            "keyword_hits": keyword_hits,
            "digit_count": digit_count,
            "line_count": line_count,
            "has_well_number": has_well_number,
            "has_date": has_date,
            "valid_ratio": valid_ratio,
            "length_score": round(length_score, 3),
            "keyword_score": round(keyword_score, 3),
            "digit_score": round(digit_score, 3),
            "structure_score": round(structure_score, 3)
        },
        "keywords_found": [kw for kw in patterns['keywords'] if kw in text]
    }


def preprocess_frame_for_ocr(frame):
    """
    预处理帧以提高OCR准确率

    Args:
        frame: numpy数组图像

    Returns:
        numpy数组: 预处理后的图像
    """
    if frame is None:
        return None

    try:
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # 直方图均衡化增强对比度
        if gray.dtype == np.uint8:
            gray = cv2.equalizeHist(gray)

        # 自适应阈值二值化
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # 轻微模糊去噪
        blurred = cv2.GaussianBlur(binary, (3, 3), 0)

        # 形态学操作（闭合小孔洞，去除小噪点）
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

        # 再次均衡化增强
        processed = cv2.equalizeHist(processed)

        return processed

    except Exception as e:
        logger.error(f"帧预处理失败: {e}")
        return frame


def enhance_ocr_accuracy(text):
    """
    后处理OCR文本，提高准确率

    Args:
        text: OCR识别文本

    Returns:
        str: 增强后的文本
    """
    if not text:
        return ""

    # 常见OCR错误修正
    replacements = {
        # 数字和字母错误
        "o": "0", "O": "0", "l": "1", "I": "1",
        # 中文错误修正
        "井号": "井号", "管逦": "管道", "检则": "检测",
        # 空格修正
        "  ": " ", "\t": " ", "  ": " "  # 重复空格
    }

    # 应用替换
    result = text
    for wrong, correct in replacements.items():
        result = result.replace(wrong, correct)

    # 规范化换行符
    result = result.replace("\r\n", "\n").replace("\r", "\n")

    # 去除重复行
    lines = result.split("\n")
    unique_lines = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped and line_stripped not in unique_lines:
            unique_lines.append(line_stripped)

    result = "\n".join(unique_lines)

    return result


def extract_pipeline_info_by_rules(text):
    """
    基于规则的管道信息提取（备用方案）

    Args:
        text: OCR识别文本

    Returns:
        dict: 提取的管道信息
    """
    if not text:
        return {
            "startWellNumber": "未知",
            "endWellNumber": "未知",
            "pipelineType": "未知",
            "pipelineMaterial": "未知",
            "inspectionDirection": "未知",
            "inspectionDate": "未知",
            "inspector": "未知"
        }

    info = {
        "startWellNumber": "未知",
        "endWellNumber": "未知",
        "pipelineType": "未知",
        "pipelineMaterial": "未知",
        "inspectionDirection": "未知",
        "inspectionDate": "未知",
        "inspector": "未知"
    }

    lines = text.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()

        # 提取井号（模式：S001, Q123等）
        well_patterns = [
            r'(起始井?号?)[:：]?\s*([SQ]\d{3,})',
            r'(终止井?号?)[:：]?\s*([SQ]\d{3,})',
            r'(起点?)[:：]?\s*([SQ]\d{3,})',
            r'(终点?)[:：]?\s*([SQ]\d{3,})',
            r'([SQ]\d{3,})'
        ]

        for pattern in well_patterns:
            matches = re.findall(pattern, line)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        label, number = match
                        if "起始" in label or "起点" in label:
                            info["startWellNumber"] = number
                        elif "终止" in label or "终点" in label:
                            info["endWellNumber"] = number
                        else:
                            # 如果没有明确标签，根据上下文判断
                            if "起始" in line or "起点" in line:
                                info["startWellNumber"] = match
                            elif "终止" in line or "终点" in line:
                                info["endWellNumber"] = match

        # 提取日期
        date_patterns = [
            r'(\d{4})[-年](\d{1,2})[-月](\d{1,2})[日]?',
            r'(\d{4})[/](\d{1,2})[/](\d{1,2})',
            r'检测日期[:：]\s*(\d{4}[-年]\d{1,2}[-月]\d{1,2}[日]?)'
        ]

        for pattern in date_patterns:
            dates = re.findall(pattern, line)
            if dates:
                for date_match in dates:
                    if isinstance(date_match, tuple):
                        year, month, day = date_match
                        info["inspectionDate"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    else:
                        info["inspectionDate"] = str(date_match)

        # 提取管道类型
        if "雨水" in line:
            info["pipelineType"] = "雨水管道"
        elif "污水" in line:
            info["pipelineType"] = "污水管道"
        elif "给水" in line:
            info["pipelineType"] = "给水管道"
        elif "排水" in line:
            info["pipelineType"] = "排水管道"

        # 提取管道材质
        materials = ["钢筋混凝土", "PVC", "HDPE", "钢管", "铸铁", "PE", "PP", "玻璃钢"]
        for material in materials:
            if material in line:
                info["pipelineMaterial"] = material
                break

        # 提取检测方向
        if "上游" in line and "下游" in line:
            info["inspectionDirection"] = "上游至下游"
        elif "北" in line and "南" in line:
            info["inspectionDirection"] = "北向南"
        elif "东" in line and "西" in line:
            info["inspectionDirection"] = "东向西"

        # 提取检测人员
        if "检测人员" in line or "检测员" in line or "操作员" in line:
            # 尝试提取姓名
            name_match = re.search(r'[:：]\s*([\u4e00-\u9fa5]{2,4})', line)
            if name_match:
                info["inspector"] = name_match.group(1)

    return info


def validate_pipeline_info(info):
    """
    验证管道信息的有效性

    Args:
        info: 管道信息字典

    Returns:
        tuple: (是否有效, 错误信息)
    """
    required_fields = ["startWellNumber", "endWellNumber", "pipelineType", "inspectionDate"]

    for field in required_fields:
        value = info.get(field, "")
        if not value or value.strip() == "" or value == "未知":
            return False, f"缺少必需字段: {field}"

    # 验证井号格式
    start_well = info.get("startWellNumber", "")
    end_well = info.get("endWellNumber", "")

    well_pattern = re.compile(r'^[A-Z]?\d+$')
    if start_well != "未知" and not well_pattern.match(start_well):
        return False, f"起始井号格式错误: {start_well}"
    if end_well != "未知" and not well_pattern.match(end_well):
        return False, f"终止井号格式错误: {end_well}"

    # 验证日期格式
    date = info.get("inspectionDate", "")
    if date != "未知":
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        if not date_pattern.match(date):
            return False, f"检测日期格式错误: {date}"

    return True, "信息有效"


def select_best_ocr_frame(frames_with_ocr, top_k=3):
    """
    从多个OCR帧中选择最佳的一帧

    Args:
        frames_with_ocr: 包含OCR结果的帧列表
        top_k: 考虑的前K个候选

    Returns:
        dict: 最佳帧信息
    """
    if not frames_with_ocr:
        return None

    # 按OCR质量评分排序
    frames_with_ocr.sort(key=lambda x: x.get("evaluation", {}).get("score", 0), reverse=True)

    # 考虑前top_k个候选
    candidates = frames_with_ocr[:min(top_k, len(frames_with_ocr))]

    # 如果有多个候选，选择综合评分最高的
    if len(candidates) >= 2:
        # 计算每个候选的详细评分
        for cand in candidates:
            score_details = cand.get("evaluation", {}).get("details", {})
            # 加权综合评分：井号存在 + 日期存在 + 文本长度
            cand["comprehensive_score"] = (
                    score_details.get("has_well_number", False) * 0.4 +
                    score_details.get("has_date", False) * 0.3 +
                    min(score_details.get("text_length", 0) / 500, 1.0) * 0.3
            )

        # 按综合评分重新排序
        candidates.sort(key=lambda x: x.get("comprehensive_score", 0), reverse=True)

    best_candidate = candidates[0]

    logger.info(f"选择最佳OCR帧: 评分{best_candidate.get('evaluation', {}).get('score', 0):.3f}, "
                f"井号{best_candidate.get('evaluation', {}).get('details', {}).get('has_well_number', False)}, "
                f"日期{best_candidate.get('evaluation', {}).get('details', {}).get('has_date', False)}")

    return best_candidate


def cleanup_temp_files():
    """清理临时文件"""
    temp_dir = tempfile.gettempdir()
    pattern = re.compile(r'^tmp.*\.(jpg|png|txt)$')

    deleted_count = 0
    for filename in os.listdir(temp_dir):
        if pattern.match(filename):
            try:
                os.unlink(os.path.join(temp_dir, filename))
                deleted_count += 1
            except Exception as e:
                logger.debug(f"删除临时文件失败 {filename}: {e}")

    if deleted_count > 0:
        logger.info(f"清理了 {deleted_count} 个临时文件")


def create_debug_directory():
    """
    创建调试目录

    Returns:
        Path: 调试目录路径
    """
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True, parents=True)

    subdirs = ["ocr_best_frames", "mirror_correction", "defect_frames"]
    for subdir in subdirs:
        (debug_dir / subdir).mkdir(exist_ok=True, parents=True)

    return debug_dir


def save_debug_info(frame, info, category="general"):
    """
    保存调试信息

    Args:
        frame: 图像帧
        info: 调试信息
        category: 分类（ocr, mirror, defect等）

    Returns:
        Path: 保存的文件路径
    """
    try:
        debug_dir = create_debug_directory()

        # 生成文件名
        timestamp = int(time.time())
        filename = f"{category}_{timestamp}_{len(info)}_{hash(str(info)) % 10000:04d}.jpg"
        filepath = debug_dir / category / filename

        # 保存图像
        cv2.imwrite(str(filepath), frame)

        # 保存信息
        info_path = filepath.with_suffix('.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.debug(f"调试信息已保存: {filepath}")
        return filepath

    except Exception as e:
        logger.warning(f"保存调试信息失败: {e}")
        return None


def format_pipeline_info_for_display(info):
    """
    格式化管道信息用于显示

    Args:
        info: 管道信息字典

    Returns:
        str: 格式化后的字符串
    """
    if not info:
        return "管道信息: 未提取"

    display_map = {
        "startWellNumber": "起始井号",
        "endWellNumber": "终止井号",
        "pipelineType": "管道类型",
        "pipelineMaterial": "管道材质",
        "inspectionDirection": "检测方向",
        "inspectionDate": "检测日期",
        "inspector": "检测人员"
    }

    lines = ["管道信息:"]
    for key, chinese_name in display_map.items():
        value = info.get(key, "未知")
        if value and value != "未知":
            lines.append(f"  {chinese_name}: {value}")

    return "\n".join(lines)


# 兼容性函数，保持原有接口
def calculate_ocr_quality(text):
    """
    兼容性函数：计算OCR质量评分

    Args:
        text: OCR识别文本

    Returns:
        float: 评分
    """
    evaluation = evaluate_ocr_quality(text)
    return evaluation["score"]