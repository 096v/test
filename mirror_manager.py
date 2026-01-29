# mirror_manager.py
import cv2
import numpy as np
import logging
import re
import tempfile
import os
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class MirrorManager:
    """
    镜像检测与纠正管理器

    自动检测视频帧是否被镜像翻转，并在需要时自动纠正
    """

    def __init__(self, ocr_func, fps: float = 30.0, config: Optional[Dict] = None):
        """
        初始化镜像管理器

        Args:
            ocr_func: OCR函数，接受图片路径或numpy数组返回文本
            fps: 视频帧率
            config: 配置字典
        """
        # 状态变量
        self.is_mirrored = False
        self.is_calibrated = False

        # 外部依赖
        self.ocr_func = ocr_func
        self.fps = fps

        # 配置
        self.config = config or {}
        calibration_seconds = self.config.get("calibration_seconds", 5)
        self.calibration_frames = int(fps * calibration_seconds)
        self.check_interval = self.config.get("ocr_check_interval", 30)
        self.energy_deviation_threshold = self.config.get("energy_deviation_threshold", 3.0)

        # 学习数据
        self.baseline_ratios = []
        self.normal_imbalance_ratio = 1.0

        # 统计信息
        self.stats = {
            'total_checks': 0,
            'mirror_flips': 0,
            'ocr_triggered': 0,
            'calibration_completed': False
        }

        # 文本验证模式
        self.valid_chars_pattern = re.compile(r'[0-9\.mMDL:()\[\]{}，。]|[a-zA-Z\u4e00-\u9fa5]')
        self.valid_numbers_pattern = re.compile(r'\d+\.?\d*')

        logger.info(f"镜像管理器初始化: 学习帧数={self.calibration_frames}, "
                    f"检查间隔={self.check_interval}, 偏离阈值={self.energy_deviation_threshold}")

    def get_edge_energy(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        计算帧边缘能量分布

        Args:
            frame: 输入图像帧

        Returns:
            tuple: (左边缘能量, 右边缘能量)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 关注UI密集的顶部和底部区域（通常是文本信息区域）
        top_height = int(h * 0.25)  # 顶部25%
        bottom_height = int(h * 0.25)  # 底部25%

        # 提取ROI
        top_roi = gray[0:top_height, :]
        bottom_roi = gray[h - bottom_height:, :]

        # 合并ROI
        if top_roi.size > 0 and bottom_roi.size > 0:
            roi = np.vstack([top_roi, bottom_roi])
        elif top_roi.size > 0:
            roi = top_roi
        elif bottom_roi.size > 0:
            roi = bottom_roi
        else:
            # 如果没有有效ROI，使用整个图像
            roi = gray

        # 使用Scharr算子获取精细边缘
        scharr_x = cv2.Scharr(roi, cv2.CV_64F, 1, 0)
        abs_scharr = np.abs(scharr_x)

        roi_w = abs_scharr.shape[1]

        # 计算左侧25%和右侧25%的能量
        left_energy = np.sum(abs_scharr[:, :roi_w // 4])
        right_energy = np.sum(abs_scharr[:, 3 * roi_w // 4:])

        # 添加微小值避免除零
        right_energy = max(right_energy, 1e-6)

        return left_energy, right_energy

    def _save_temp_image(self, frame: np.ndarray) -> str:
        """
        保存临时图像文件

        Args:
            frame: 图像帧

        Returns:
            str: 临时文件路径
        """
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = tmp_file.name
            cv2.imwrite(temp_path, frame)
            return temp_path

    def _calculate_ocr_score(self, text: str) -> float:
        """
        计算OCR文本的合法性得分

        Args:
            text: OCR识别文本

        Returns:
            float: 得分
        """
        if not text:
            return 0.0

        # 有效字符比例
        valid_chars = self.valid_chars_pattern.findall(text)
        valid_ratio = len(valid_chars) / len(text) if text else 0

        # 数字密度
        numbers = self.valid_numbers_pattern.findall(text)
        number_density = len(numbers) / max(len(text.split()), 1)

        # 关键词检测
        keywords = ['管道', '井号', '距离', '管径', 'mm', 'm', '检测', '深度', '井', '管']
        keyword_score = sum(1 for kw in keywords if kw in text)

        # 综合得分
        total_score = valid_ratio * 40 + number_density * 30 + keyword_score * 30

        return total_score

    def check_ocr_validity(self, frame: np.ndarray) -> bool:
        """
        通过OCR验证镜像状态的合法性

        Args:
            frame: 图像帧（numpy数组）

        Returns:
            bool: 是否需要镜像翻转
        """
        self.stats['ocr_triggered'] += 1

        try:
            # 测试原图OCR - 直接传入numpy数组
            text_orig = self.ocr_func(frame)
            score_orig = self._calculate_ocr_score(text_orig)

            # 生成镜像图并测试
            flipped_frame = cv2.flip(frame, 1)
            text_flipped = self.ocr_func(flipped_frame)
            score_flipped = self._calculate_ocr_score(text_flipped)

            logger.debug(f"OCR验证: 原方向={score_orig:.1f}, 镜像方向={score_flipped:.1f}")

            # 镜像得分明显更高时返回True
            return score_flipped > (score_orig + 2)

        except Exception as e:
            logger.warning(f"OCR验证失败: {e}")
            return False

    def update_and_correct(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        更新镜像状态并返回纠正后的帧

        Args:
            frame: 输入图像帧
            frame_idx: 当前帧索引

        Returns:
            np.ndarray: 纠正后的图像帧
        """
        self.stats['total_checks'] += 1

        # 1. 计算边缘能量
        left_energy, right_energy = self.get_edge_energy(frame)
        current_ratio = left_energy / right_energy

        # 2. 学习阶段（前N帧）
        if not self.is_calibrated:
            if frame_idx < self.calibration_frames:
                self.baseline_ratios.append(current_ratio)
                if frame_idx % 30 == 0:
                    logger.info(f"学习视频布局特征... ({frame_idx}/{self.calibration_frames})")
                return frame
            else:
                # 学习完成，计算基准比值
                self.normal_imbalance_ratio = np.median(self.baseline_ratios)
                self.is_calibrated = True
                self.stats['calibration_completed'] = True

                # 统计信息
                min_ratio = np.min(self.baseline_ratios)
                max_ratio = np.max(self.baseline_ratios)
                std_ratio = np.std(self.baseline_ratios)

                logger.info(f"学习完成！基准能量比: {self.normal_imbalance_ratio:.3f} "
                            f"(范围: {min_ratio:.3f}~{max_ratio:.3f}, 标准差: {std_ratio:.3f})")
                return frame

        # 3. 监测阶段（学习完成后）
        if frame_idx % self.check_interval == 0:
            # 计算偏离度
            if self.normal_imbalance_ratio != 0:
                deviation = current_ratio / self.normal_imbalance_ratio
            else:
                deviation = current_ratio

            # 判断是否需要检查
            needs_check = False
            threshold = self.energy_deviation_threshold

            if not self.is_mirrored:
                # 如果偏离度超过阈值，说明可能镜像了
                if deviation > threshold or deviation < (1 / threshold):
                    needs_check = True
                    logger.debug(f"帧{frame_idx}: 显著布局偏离 (偏离度={deviation:.2f})")
            else:
                # 如果已经镜像，检查是否应该恢复
                if 0.8 < deviation < 1.25:  # 接近正常范围
                    needs_check = True
                    logger.debug(f"帧{frame_idx}: 布局接近正常 (偏离度={deviation:.2f})")

            # 如果需要检查，使用OCR确认
            if needs_check:
                if self.check_ocr_validity(frame):
                    self.is_mirrored = not self.is_mirrored
                    self.stats['mirror_flips'] += 1
                    status = "镜像" if self.is_mirrored else "正常"
                    logger.warning(f"帧{frame_idx}: 镜像状态切换 -> {status}模式")

        # 4. 执行纠正
        if self.is_mirrored:
            corrected_frame = cv2.flip(frame, 1)
            if frame_idx % 100 == 0:
                logger.debug(f"帧{frame_idx}: 应用镜像纠正")
            return corrected_frame

        return frame

    def get_status(self) -> Dict[str, Any]:
        """
        获取镜像管理器状态

        Returns:
            dict: 状态信息
        """
        return {
            'is_mirrored': self.is_mirrored,
            'is_calibrated': self.is_calibrated,
            'normal_imbalance_ratio': self.normal_imbalance_ratio if self.is_calibrated else None,
            'stats': self.stats.copy(),
            'config': {
                'calibration_frames': self.calibration_frames,
                'check_interval': self.check_interval,
                'energy_deviation_threshold': self.energy_deviation_threshold
            }
        }

    def reset(self):
        """重置镜像管理器状态"""
        self.is_mirrored = False
        self.is_calibrated = False
        self.baseline_ratios = []
        self.normal_imbalance_ratio = 1.0
        self.stats = {
            'total_checks': 0,
            'mirror_flips': 0,
            'ocr_triggered': 0,
            'calibration_completed': False
        }
        logger.info("镜像管理器已重置")