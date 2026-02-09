import os
import sys
import time
import argparse
import logging
import json
import cv2
import base64
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# 导入自定义模块
from config_loader import load_config, create_config_file, validate_config
from mirror_manager import MirrorManager
from utils import get_ai_summary_api, paddle_ocr_api, NumpyEncoder


# ----------------------------------------------------------------------
# 日志配置
# ----------------------------------------------------------------------
class SafeStreamHandler(logging.StreamHandler):
    """解决Windows控制台编码问题的Handler"""

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                if hasattr(stream, 'encoding') and stream.encoding:
                    stream.write(msg + self.terminator)
                else:
                    stream.write((msg + self.terminator).encode('utf-8', errors='replace').decode('utf-8'))
            except Exception:
                stream.write((msg + self.terminator).encode('utf-8', errors='replace').decode('utf-8'))
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler('defect_detection.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = SafeStreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()


# ----------------------------------------------------------------------
# OCR帧预处理函数
# ----------------------------------------------------------------------
def preprocess_frame_for_ocr(frame):
    """
    简化预处理：主要解决颜色空间问题，避免过度处理

    Args:
        frame: numpy数组图像

    Returns:
        numpy数组: 预处理后的RGB图像
    """
    if frame is None:
        return None

    try:
        # 确保是彩色图像（PaddleOCR对彩色图像识别更好）
        if len(frame.shape) == 3:
            # OpenCV读取的是BGR，转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 可选：增强对比度（但不过度处理）
            # 转换为HSV空间，调整V通道
            hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)

            # 直方图均衡化增强对比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v_eq = clahe.apply(v)

            # 合并通道并转回RGB
            hsv_eq = cv2.merge([h, s, v_eq])
            enhanced = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

            return enhanced
        else:
            # 如果是灰度图，转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            return rgb_frame

    except Exception as e:
        logger.error(f"帧预处理失败: {e}")
        # 出错时返回原始帧（确保是RGB）
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return frame


# 在 main.py 中修改 evaluate_ocr_quality 函数
def evaluate_ocr_quality(text, patterns=None):
    """
    评估OCR文本质量，返回评分和细节

    Args:
        text: OCR识别文本
        patterns: 自定义模式字典

    Returns:
        dict: 包含评分和细节的字典
    """
    if not text or text.strip() == "":
        return {"score": 0.0, "details": {}, "keywords_found": []}

    text = text.strip()
    text_length = len(text)

    # 如果文本太短，直接返回低分
    if text_length < 5:
        return {"score": 0.1, "details": {"text_length": text_length}, "keywords_found": []}

    # 默认管道检测关键词
    if patterns is None:
        patterns = {
            'keywords': [
                "井", "雨", "污", "管", "道", "检测",
                "S", "Q", "Φ", "DN", "编号", "人员",
                "雨水", "污水", "给水", "管道", "井号",
                "材质", "方向", "检测井", "检查井"
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

    import re

    # 1. 文本长度得分
    max_expected_len = 1000
    length_score = min(text_length / max_expected_len, 1.0)

    # 2. 关键词匹配得分
    keywords_found = []
    for kw in patterns['keywords']:
        if kw in text:
            keywords_found.append(kw)

    keyword_hits = len(keywords_found)
    keyword_score = min(keyword_hits / 10, 1.0)  # 最多10个关键词，超过也算满分

    # 3. 数字密度得分（管道检测中数字很重要）
    digit_count = sum(c.isdigit() for c in text)
    digit_score = min(digit_count / max(text_length, 1) * 20, 1.0)

    # 4. 检查是否有井号、日期等关键信息
    has_well_number = any(re.search(pattern, text) for pattern in patterns['well_patterns'])
    has_date = any(re.search(pattern, text) for pattern in patterns['date_patterns'])

    structure_score = 0.0
    if has_well_number:
        structure_score += 0.5
    if has_date:
        structure_score += 0.5

    # 5. 有效字符比例（中文、数字、常见符号）
    valid_chars = re.findall(r'[0-9a-zA-Z\u4e00-\u9fa5，。、；：！？（）《》【】·\-\./:]', text)
    valid_ratio = len(valid_chars) / max(text_length, 1)

    # 计算加权总分（调整权重，更注重关键词和结构）
    weights = {
        'length': 0.10,
        'keyword': 0.50,  # 关键词权重最高
        'digit': 0.15,
        'structure': 0.20,
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
            "has_well_number": has_well_number,
            "has_date": has_date,
            "valid_ratio": valid_ratio,
            "length_score": round(length_score, 3),
            "keyword_score": round(keyword_score, 3),
            "digit_score": round(digit_score, 3),
            "structure_score": round(structure_score, 3)
        },
        "keywords_found": keywords_found
    }


# ----------------------------------------------------------------------
# 核心检测类
# ----------------------------------------------------------------------
class DefectDetector:
    def __init__(self, video_path=None, model_path=None, config_path=None, no_mirror=False):
        # 1. 加载配置
        self.config, _ = load_config(config_path)
        if not validate_config(self.config)[0]:
            raise ValueError("配置文件验证失败")

        self.video_path = video_path or self.config["video_path"]
        self.model_path = model_path or self.config["model_path"]

        # 2. 加载模型
        logger.info(f"加载YOLO模型: {self.model_path}")
        self.model = YOLO(self.model_path)

        # 3. 打开视频
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.current_frame_idx = 0
        self.detection_start_time = None

        # 4. 初始化组件
        self.frame_save_dir = Path(self.config["frame_save_dir"])
        self.frame_save_dir.mkdir(exist_ok=True)

        # 镜像管理器
        self.mirror_enabled = self.config.get("mirror_detection", {}).get("enabled", True) and not no_mirror
        if self.mirror_enabled:
            self.mirror_manager = MirrorManager(
                ocr_func=lambda img: paddle_ocr_api(img, self.config["paddle_ocr"]),
                fps=self.fps,
                config=self.config.get("mirror_detection", {})
            )
        else:
            self.mirror_manager = None

        # 5. 追踪状态变量
        self.active_tracks = {}
        self.completed_tracks = []
        self.track_type_counter = {}  # 统计每个ID的类型投票 {id: {'TL': 10, 'CR': 2}}

        # 添加：用于缓存开始帧和结束帧的图片
        self.frame_cache = {}  # 缓存最近处理的帧 {frame_idx: frame_image}
        self.max_cache_size = 1000  # 最大缓存帧数

        # 配置参数
        self.max_lost_frames = self.config.get("max_lost_frames", 30)
        self.min_duration_frames = self.config.get("min_duration_frames", 10)
        self.base64_quality = self.config.get("base64_encoding", {}).get("quality", 80)

        # OCR提取配置
        self.ocr_extraction_config = self.config.get("ocr_extraction", {})
        self.sample_frames = self.ocr_extraction_config.get("sample_frames", 30)
        self.min_score_threshold = self.ocr_extraction_config.get("min_score_threshold", 0.3)

        # 6. 执行优化后的管道信息提取策略
        self.global_pipeline_info, self.ocr_statistics = self.extract_pipeline_info_optimized()

        logger.info(f"管道信息提取完成: {self.global_pipeline_info}")
        logger.info(f"OCR统计: 最佳帧评分={self.ocr_statistics.get('best_frame_score', 0):.3f}, "
                    f"方法={self.ocr_statistics.get('extraction_method', 'unknown')}")

    def _cache_frame(self, frame_idx, frame):
        """缓存帧图片"""
        if len(self.frame_cache) >= self.max_cache_size:
            # 移除最旧的帧
            oldest_key = min(self.frame_cache.keys())
            del self.frame_cache[oldest_key]

        # 压缩存储（可选）
        if self.frame_width > 800:
            h, w = frame.shape[:2]
            scale = 800 / w
            small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            self.frame_cache[frame_idx] = small_frame
        else:
            self.frame_cache[frame_idx] = frame

    def _get_cached_frame(self, frame_idx):
        """获取缓存的帧"""
        if frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx].copy()
        return None

    def _clean_frame_cache(self):
        """清理过期的缓存（保留最近1000帧）"""
        if len(self.frame_cache) > self.max_cache_size:
            # 保留最近的帧
            current_frame = self.current_frame_idx
            frames_to_remove = []
            for frame_idx in self.frame_cache.keys():
                if current_frame - frame_idx > self.max_cache_size:
                    frames_to_remove.append(frame_idx)

            for frame_idx in frames_to_remove:
                del self.frame_cache[frame_idx]

    def _encode_frame_to_base64(self, frame):
        """将帧转换为Base64字符串"""
        if frame is None:
            return None
        try:
            # 缩放以减小体积
            h, w = frame.shape[:2]
            if w > 800:
                scale = 800 / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.base64_quality])
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Base64编码失败: {e}")
            return None

    def extract_pipeline_info_optimized(self):
        """
        优化的管道信息提取策略：前30帧采样评分

        策略步骤：
        1. 采样前30帧（或按间隔采样）
        2. 对每帧进行预处理和OCR
        3. 使用评分机制选择最佳帧
        4. 使用AI提取结构化信息
        5. 多层验证和备选方案
        """
        logger.info("开始优化版管道信息提取（前30帧采样评分策略）...")

        # 保存原始视频位置
        original_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        try:
            # 1. 采样和评估帧
            candidates = self.sample_and_evaluate_frames(max_frames=self.sample_frames)

            if not candidates:
                logger.warning("未找到合适的OCR候选帧，使用默认值")
                return self._get_default_pipeline_info(), {"extraction_method": "default", "sampled_frames": 0}

            # 2. 选择最佳帧
            best_candidate = self.select_best_candidate(candidates)

            if not best_candidate or best_candidate["evaluation"]["score"] < self.min_score_threshold:
                logger.warning(
                    f"最佳帧评分过低({best_candidate['evaluation']['score'] if best_candidate else 0:.3f})，使用合并策略")
                pipeline_info = self._extract_from_multiple_candidates(candidates)
                return pipeline_info, {
                    "extraction_method": "multi_frame_merge",
                    "sampled_frames": len(candidates),
                    "best_frame_score": best_candidate["evaluation"]["score"] if best_candidate else 0,
                    "merge_count": min(5, len(candidates))
                }

            # 3. 从最佳帧提取信息
            pipeline_info = self._extract_from_best_candidate(best_candidate)

            # 4. 验证提取的信息
            if self._validate_pipeline_info(pipeline_info):
                logger.info(
                    f"成功从帧 {best_candidate['frame_num']} 提取管道信息，评分: {best_candidate['evaluation']['score']:.3f}")

                # 保存最佳帧用于调试
                if self.ocr_extraction_config.get("debug_mode", False):
                    self._save_best_frame_for_debug(best_candidate)

                return pipeline_info, {
                    "extraction_method": "best_frame",
                    "sampled_frames": len(candidates),
                    "best_frame_score": best_candidate["evaluation"]["score"],
                    "best_frame_num": best_candidate["frame_num"],
                    "keywords_found": len(best_candidate["evaluation"]["keywords_found"])
                }
            else:
                logger.warning("提取的管道信息验证失败，使用合并策略")
                pipeline_info = self._extract_from_multiple_candidates(candidates[:5])
                return pipeline_info, {
                    "extraction_method": "multi_frame_merge_after_validation_failed",
                    "sampled_frames": len(candidates),
                    "best_frame_score": best_candidate["evaluation"]["score"],
                    "merge_count": min(5, len(candidates))
                }

        except Exception as e:
            logger.error(f"优化提取策略失败: {e}")
            return self._get_default_pipeline_info(), {"extraction_method": "error", "error": str(e)}
        finally:
            # 恢复视频位置
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

    def sample_and_evaluate_frames(self, max_frames=30):
        """
        采样并评估视频帧

        Args:
            max_frames: 最大采样帧数

        Returns:
            list: 候选帧列表
        """
        candidates = []
        total_frames = self.total_frames

        # 计算采样间隔（均匀采样）
        interval = max(1, total_frames // max_frames)
        sample_points = list(range(0, min(total_frames, max_frames * interval), interval))

        logger.info(f"视频总帧数: {total_frames}, 采样间隔: {interval}, 采样点: {len(sample_points)}")

        for i, frame_idx in enumerate(sample_points[:max_frames]):
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()

                if not ret:
                    logger.warning(f"无法读取帧 {frame_idx}")
                    continue

                logger.debug(f"处理帧 {frame_idx}: 原始帧大小={frame.shape}")

                # 预处理帧（简化处理）
                processed_frame = preprocess_frame_for_ocr(frame)

                # 调试：保存原始帧和预处理帧
                if self.ocr_extraction_config.get("debug_mode", False):
                    debug_dir = Path("debug/frames")
                    debug_dir.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(debug_dir / f"raw_{frame_idx:04d}.jpg"), frame)
                    cv2.imwrite(str(debug_dir / f"processed_{frame_idx:04d}.jpg"),
                                cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

                # OCR识别
                ocr_text = paddle_ocr_api(processed_frame, self.config["paddle_ocr"])

                # 记录OCR结果（即使是空文本）
                if ocr_text and len(ocr_text.strip()) > 0:
                    logger.debug(f"帧 {frame_idx} OCR结果: {ocr_text[:100]}...")
                else:
                    logger.warning(f"帧 {frame_idx} OCR结果为空")

                # 评估OCR质量
                evaluation = evaluate_ocr_quality(ocr_text)

                candidate = {
                    "frame_num": frame_idx,
                    "frame_time": frame_idx / self.fps if self.fps > 0 else 0,
                    "raw_frame": frame.copy(),
                    "processed_frame": processed_frame,
                    "ocr_text": ocr_text,
                    "evaluation": evaluation
                }

                candidates.append(candidate)

                # 如果找到高质量帧，可以提前终止
                if evaluation["score"] > 0.6 and evaluation["details"]["has_well_number"]:
                    logger.info(f"找到高质量帧 {frame_idx}，提前终止采样")
                    break

            except Exception as e:
                logger.error(f"处理帧 {frame_idx} 时出错: {e}", exc_info=True)
                continue

        # 按评分排序
        candidates.sort(key=lambda x: x["evaluation"]["score"], reverse=True)

        # 记录前5个候选帧信息
        for i, cand in enumerate(candidates[:5]):
            logger.info(f"候选帧 {i + 1}: 帧{cand['frame_num']}, "
                        f"评分{cand['evaluation']['score']:.3f}, "
                        f"关键词{len(cand['evaluation']['keywords_found'])}个, "
                        f"文本长度{len(cand['ocr_text'])}")

        if candidates and candidates[0]["evaluation"]["score"] > 0:
            logger.info(f"最佳OCR文本(前100字符): {candidates[0]['ocr_text'][:100]}...")

        return candidates

    def select_best_candidate(self, candidates):
        """
        从候选帧中选择最佳的一帧

        Args:
            candidates: 候选帧列表

        Returns:
            dict: 最佳候选帧
        """
        if not candidates:
            return None

        # 取评分最高的帧
        best = candidates[0]

        # 如果前几帧评分相近，选择较早的帧（通常管道信息在开头）
        if len(candidates) >= 3:
            top3 = candidates[:3]
            scores = [c["evaluation"]["score"] for c in top3]

            # 如果评分差异小于0.1，选择最早出现的帧
            if max(scores) - min(scores) < 0.1:
                best = min(top3, key=lambda x: x["frame_num"])
                logger.info(f"前3帧评分相近，选择较早的帧 {best['frame_num']}")

        return best

    def _extract_from_best_candidate(self, best_candidate):
        """从最佳候选帧提取管道信息"""
        ocr_text = best_candidate["ocr_text"]

        # 调用AI提取结构化信息
        ai_result = get_ai_summary_api(ocr_text, self.config["ai_service"])

        # 确保返回7个字段
        return self._ensure_seven_fields(ai_result)

    def _extract_from_multiple_candidates(self, candidates, top_k=5):
        """
        从多个候选帧合并提取管道信息

        Args:
            candidates: 候选帧列表
            top_k: 使用前k个候选帧

        Returns:
            dict: 合并提取的管道信息
        """
        # 合并前k个候选帧的文本
        texts_to_merge = []
        for cand in candidates[:top_k]:
            if cand["ocr_text"] and len(cand["ocr_text"]) > 10:
                texts_to_merge.append(cand["ocr_text"])

        if not texts_to_merge:
            return self._get_default_pipeline_info()

        # 如果有多个文本，合并并去重
        merged_text = "\n---\n".join(texts_to_merge)

        # 添加提示说明这是合并文本
        prompt_addition = "注意：这是从视频的多个帧中提取的OCR文本，请综合分析提取最可靠的信息。"
        enhanced_text = f"{prompt_addition}\n\n{merged_text}"

        # 调用AI提取
        ai_result = get_ai_summary_api(enhanced_text, self.config["ai_service"])

        if isinstance(ai_result, dict):
            return self._ensure_seven_fields(ai_result)

        return self._get_default_pipeline_info()

    def _save_best_frame_for_debug(self, best_candidate):
        """保存最佳帧用于调试"""
        try:
            debug_dir = Path("debug/ocr_best_frames")
            debug_dir.mkdir(exist_ok=True, parents=True)

            frame_num = best_candidate["frame_num"]
            score = best_candidate["evaluation"]["score"]

            # 保存原始帧
            raw_path = debug_dir / f"best_raw_{frame_num:04d}_score{score:.2f}.jpg"
            cv2.imwrite(str(raw_path), best_candidate["raw_frame"])

            # 保存预处理后的帧
            processed_path = debug_dir / f"best_processed_{frame_num:04d}_score{score:.2f}.jpg"
            cv2.imwrite(str(processed_path), best_candidate["processed_frame"])

            # 保存OCR文本和评分
            info_path = debug_dir / f"best_info_{frame_num:04d}_score{score:.2f}.txt"
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"帧号: {frame_num}\n")
                f.write(f"评分: {score:.3f}\n")
                f.write(f"关键词: {', '.join(best_candidate['evaluation']['keywords_found'])}\n")
                f.write("\n" + "=" * 50 + "\n")
                f.write("OCR文本:\n")
                f.write(best_candidate["ocr_text"])

            logger.info(f"最佳帧调试信息已保存到: {debug_dir}")

        except Exception as e:
            logger.warning(f"保存最佳帧调试信息失败: {e}")

    def _validate_pipeline_info(self, info):
        """验证管道信息的有效性"""
        required_fields = ["startWellNumber", "endWellNumber", "pipelineType", "inspectionDate"]

        for field in required_fields:
            value = info.get(field, "")
            if not value or value.strip() == "" or value == "未知":
                return False

        return True

    def _ensure_seven_fields(self, info):
        """确保返回7个字段"""
        default = self._get_default_pipeline_info()

        if not isinstance(info, dict):
            return default

        # 合并AI结果和默认值
        result = default.copy()

        # 更新非空值
        for key in result.keys():
            if key in info and info[key] and str(info[key]).strip():
                result[key] = str(info[key]).strip()

        return result

    def _get_default_pipeline_info(self):
        """返回默认的管道信息结构"""
        return {
            "startWellNumber": "未知",
            "endWellNumber": "未知",
            "pipelineType": "未知",
            "pipelineMaterial": "未知",
            "inspectionDirection": "未知",
            "inspectionDate": "未知",
            "inspector": "未知"
        }

    def process_frame(self, frame):
        """处理单帧：镜像纠正 -> YOLO -> 轨迹更新"""
        # 缓存当前帧，用于后续可能需要的开始帧/结束帧提取
        self._cache_frame(self.current_frame_idx, frame.copy())

        # 1. 镜像纠正
        if self.mirror_manager:
            frame = self.mirror_manager.update_and_correct(frame, self.current_frame_idx)

        # 2. YOLO 追踪
        results = self.model.track(
            source=frame, persist=True, tracker="bytetrack.yaml",
            conf=self.config["yolo_confidence"], iou=self.config["yolo_iou"], verbose=False
        )

        detected_ids = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            cls_ids = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            names = self.model.names

            for box, tid, cid, conf in zip(boxes, track_ids, cls_ids, confs):
                detected_ids.append(tid)
                label = names[cid]

                # 初始化轨迹
                if tid not in self.active_tracks:
                    self.active_tracks[tid] = {
                        "track_id": int(tid),
                        "defect_type": label,  # 初始类型
                        "start_frame": self.current_frame_idx,
                        "end_frame": self.current_frame_idx,
                        "start_time": self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                        "max_conf": float(conf),
                        "frames_detected": 0,
                        # 保存开始帧图片（使用缓存）
                        "start_frame_img": self._get_cached_frame(self.current_frame_idx),
                        # 初始化结束帧图片（后续会更新）
                        "end_frame_img": None,
                        "last_seen": self.current_frame_idx
                    }
                    self.track_type_counter[tid] = defaultdict(int)

                # 更新轨迹
                track = self.active_tracks[tid]
                track["end_frame"] = self.current_frame_idx
                track["end_time"] = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                track["last_seen"] = self.current_frame_idx
                track["frames_detected"] += 1

                # 更新结束帧图片
                track["end_frame_img"] = frame.copy()

                # 更新类型统计
                self.track_type_counter[tid][label] += 1

                # 更新最大置信度
                if conf > track["max_conf"]:
                    track["max_conf"] = float(conf)

        # 3. 检查丢失的轨迹
        ids_to_remove = []
        for tid, track in self.active_tracks.items():
            if self.current_frame_idx - track["last_seen"] > self.max_lost_frames:
                self.settle_track(tid)
                ids_to_remove.append(tid)

        for tid in ids_to_remove:
            del self.active_tracks[tid]

        # 清理过期的缓存
        self._clean_frame_cache()

        return results[0].plot() if results else frame

    def settle_track(self, tid):
        """结算轨迹：计算最终类型、生成开始帧和结束帧Base64"""
        track = self.active_tracks[tid]
        duration = track["end_frame"] - track["start_frame"] + 1

        if duration < self.min_duration_frames:
            return  # 过滤短时误检

        # 多数投票决定类型
        type_stats = self.track_type_counter.get(tid, {})
        if type_stats:
            final_type = max(type_stats.items(), key=lambda x: x[1])[0]
            total_votes = sum(type_stats.values())
            percentage = (type_stats[final_type] / total_votes * 100) if total_votes > 0 else 0.0
        else:
            final_type = track["defect_type"]
            percentage = 100.0

        # 生成开始帧和结束帧的Base64
        start_frame_img = track.get("start_frame_img")
        end_frame_img = track.get("end_frame_img")

        # 如果开始帧图片为空，尝试从缓存获取
        if start_frame_img is None:
            start_frame_img = self._get_cached_frame(track["start_frame"])

        start_base64 = self._encode_frame_to_base64(start_frame_img)
        end_base64 = self._encode_frame_to_base64(end_frame_img)

        # 构建完成的轨迹对象
        completed_track = {
            "track_id": track["track_id"],
            "defect_type": final_type,
            "start_frame": track["start_frame"],
            "end_frame": track["end_frame"],
            "duration_frames": duration,
            "start_time": round(track["start_time"], 1),
            "end_time": round(track["end_time"], 1),
            "duration_seconds": round(track["end_time"] - track["start_time"], 1),
            "max_confidence": round(track["max_conf"], 3),
            "frames_detected": track["frames_detected"],
            "type_percentage": round(percentage, 1),
            "type_statistics": dict(type_stats),
            "mirror_correction": None,  # 显式 Null

            # 两张图片：开始帧和结束帧
            "start_frame_image_base64": start_base64,
            "end_frame_image_base64": end_base64
        }

        self.completed_tracks.append(completed_track)

        # 清理内存
        if tid in self.track_type_counter:
            del self.track_type_counter[tid]

        # 清理轨迹中的图片引用，释放内存
        if "start_frame_img" in self.active_tracks[tid]:
            del self.active_tracks[tid]["start_frame_img"]
        if "end_frame_img" in self.active_tracks[tid]:
            del self.active_tracks[tid]["end_frame_img"]

    def generate_report(self):
        """生成最终 JSON 报告"""
        # 结算剩余轨迹
        for tid in list(self.active_tracks.keys()):
            self.settle_track(tid)

        # 统计数据
        defect_stats = defaultdict(int)
        for t in self.completed_tracks:
            defect_stats[t["defect_type"]] += 1

        total_time = time.time() - self.detection_start_time

        # 构造最终 JSON
        report = {
            "total_frames": self.total_frames,
            "processed_frames": self.current_frame_idx,
            "detection_time": round(total_time, 2),
            "result_video_url": "",
            # 严格使用 7 项 Pipeline Info
            "pipeline_info": self.global_pipeline_info,
            "ocr_statistics": self.ocr_statistics,  # 添加OCR统计信息
            "mirror_correction": {
                "enabled": self.mirror_enabled,
                "is_mirrored": self.mirror_manager.is_mirrored if self.mirror_manager else False
            },
            "detection_summary": {
                "total_tracks": len(self.completed_tracks),
                "defect_type_stats": dict(defect_stats),
                "detection_config": {
                    "confidence_threshold": self.config["yolo_confidence"],
                    "iou_threshold": self.config["yolo_iou"],
                    "min_duration_frames": self.min_duration_frames,
                    "max_lost_frames": self.max_lost_frames,
                    "ocr_extraction_method": self.ocr_statistics.get("extraction_method", "unknown")
                }
            },
            # 按开始帧排序
            "tracks": sorted(self.completed_tracks, key=lambda x: x["start_frame"])
        }

        # 保存
        out_path = self.config.get("output_report", "defect_report.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        logger.info(f"报告已生成: {out_path}")
        return report

    def run(self, show=False):
        self.detection_start_time = time.time()
        logger.info("开始处理视频...")

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break

                annotated_frame = self.process_frame(frame)
                self.current_frame_idx += 1

                if show:
                    cv2.imshow("Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break

                if self.current_frame_idx % 50 == 0:
                    logger.info(f"处理进度: {self.current_frame_idx}/{self.total_frames}")

        except Exception as e:
            logger.error(f"处理中断: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.generate_report()


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='视频路径')
    parser.add_argument('--config', type=str, default='config.json', help='配置路径')
    parser.add_argument('--create-config', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--no-mirror', action='store_true')
    args = parser.parse_args()

    if args.create_config:
        create_config_file(args.config)
        return

    if not os.path.exists(args.config):
        print("配置文件不存在，请先运行 --create-config")
        return

    try:
        detector = DefectDetector(
            video_path=args.video,
            config_path=args.config,
            no_mirror=args.no_mirror
        )
        detector.run(show=args.show)
    except Exception as e:
        logger.critical(f"程序崩溃: {e}")


if __name__ == "__main__":
    main()