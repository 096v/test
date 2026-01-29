# config_loader.py
import json
import logging
import os

logger = logging.getLogger(__name__)

# 默认配置 - 更新OCR配置部分
DEFAULT_CONFIG = {
    "video_path": "MyVideo_1.mp4",
    "model_path": "weights/best.pt",
    "frame_save_dir": "frames",
    "output_report": "defect_report.json",
    "max_lost_frames": 30,
    "min_duration_frames": 30,
    "yolo_confidence": 0.3,
    "yolo_iou": 0.5,
    "mirror_detection": {
        "enabled": True,
        "calibration_seconds": 5,
        "energy_deviation_threshold": 3.0,
        "ocr_check_interval": 30
    },
    "paddle_ocr": {
        "use_angle_cls": True,
        "lang": "ch",
        "use_gpu": False,
        "gpu_mem": 500,
        "cpu_threads": 10,
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.6,
        "drop_score": 0.5,
        "min_confidence": 0.5,
        "show_log": False
    },
    "ai_service": {
        "url": "http://127.0.0.1:11434/api/generate",
        "model": "gemma3:4b",
        "timeout": 120
    }
}

DEFAULT_CONFIG_FILE = "config.json"


def load_config(config_path=None):
    """
    加载配置，优先使用环境变量，其次使用配置文件

    Args:
        config_path: 配置文件路径，默认使用 DEFAULT_CONFIG_FILE

    Returns:
        tuple: (配置字典, 配置文件路径)
    """
    config = DEFAULT_CONFIG.copy()

    # 确定配置文件路径
    config_file_path = config_path or DEFAULT_CONFIG_FILE

    # 如果配置文件存在，加载配置
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config.update(file_config)
            logger.info(f"从 {config_file_path} 加载配置成功")
        except json.JSONDecodeError as e:
            logger.error(f"配置文件 JSON 格式错误: {e}")
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}")
    else:
        logger.warning(f"配置文件 {config_file_path} 不存在，使用默认配置")

    # 环境变量覆盖（用于安全部署）
    env_mappings = {
        # PaddleOCR相关配置
        "PADDLE_OCR_USE_ANGLE_CLS": ("paddle_ocr", "use_angle_cls"),
        "PADDLE_OCR_LANG": ("paddle_ocr", "lang"),
        "PADDLE_OCR_USE_GPU": ("paddle_ocr", "use_gpu"),
        "PADDLE_OCR_GPU_MEM": ("paddle_ocr", "gpu_mem"),
        "PADDLE_OCR_CPU_THREADS": ("paddle_ocr", "cpu_threads"),
        "PADDLE_OCR_DET_DB_THRESH": ("paddle_ocr", "det_db_thresh"),
        "PADDLE_OCR_DET_DB_BOX_THRESH": ("paddle_ocr", "det_db_box_thresh"),
        "PADDLE_OCR_DROP_SCORE": ("paddle_ocr", "drop_score"),
        "PADDLE_OCR_MIN_CONFIDENCE": ("paddle_ocr", "min_confidence"),
        "PADDLE_OCR_SHOW_LOG": ("paddle_ocr", "show_log"),

        # AI服务配置
        "AI_SERVICE_URL": ("ai_service", "url"),
        "AI_MODEL": ("ai_service", "model"),
        "AI_TIMEOUT": ("ai_service", "timeout"),

        # 其他配置
        "VIDEO_PATH": ("video_path", None),
        "MODEL_PATH": ("model_path", None),
        "MIRROR_DETECTION_ENABLED": ("mirror_detection", "enabled"),
        "MIRROR_CALIBRATION_SECONDS": ("mirror_detection", "calibration_seconds"),
        "MIRROR_DEVIATION_THRESHOLD": ("mirror_detection", "energy_deviation_threshold"),
        "MIRROR_CHECK_INTERVAL": ("mirror_detection", "ocr_check_interval"),
        "YOLO_CONFIDENCE": ("yolo_confidence", None),
        "YOLO_IOU": ("yolo_iou", None),
        "MAX_LOST_FRAMES": ("max_lost_frames", None),
        "MIN_DURATION_FRAMES": ("min_duration_frames", None)
    }

    for env_key, config_path_tuple in env_mappings.items():
        env_value = os.getenv(env_key)
        if env_value is not None:
            section, key = config_path_tuple
            if key:  # 嵌套配置
                # 处理布尔值
                if env_key in ["PADDLE_OCR_USE_ANGLE_CLS", "PADDLE_OCR_USE_GPU",
                               "PADDLE_OCR_SHOW_LOG", "MIRROR_DETECTION_ENABLED"]:
                    config[section][key] = env_value.lower() in ('true', '1', 'yes', 't')
                # 处理数值类型
                elif env_key in ["PADDLE_OCR_GPU_MEM", "PADDLE_OCR_CPU_THREADS",
                                 "PADDLE_OCR_DET_DB_THRESH", "PADDLE_OCR_DET_DB_BOX_THRESH",
                                 "PADDLE_OCR_DROP_SCORE", "PADDLE_OCR_MIN_CONFIDENCE",
                                 "MIRROR_CALIBRATION_SECONDS", "MIRROR_DEVIATION_THRESHOLD",
                                 "MIRROR_CHECK_INTERVAL", "YOLO_CONFIDENCE", "YOLO_IOU",
                                 "MAX_LOST_FRAMES", "MIN_DURATION_FRAMES", "AI_TIMEOUT"]:
                    try:
                        # 尝试转换为浮点数
                        if "." in env_value:
                            config[section][key] = float(env_value)
                        else:
                            config[section][key] = int(env_value)
                    except ValueError:
                        config[section][key] = env_value
                else:
                    config[section][key] = env_value
            else:  # 顶级配置
                config[section] = env_value

    return config, config_file_path


def create_config_file(config_path=None):
    """
    创建配置文件模板

    Args:
        config_path: 配置文件路径

    Returns:
        bool: 是否成功创建
    """
    config_file = config_path or DEFAULT_CONFIG_FILE

    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)

    if not os.path.exists(config_file):
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
            logger.info(f"已创建配置文件: {config_file}")
            return True
        except Exception as e:
            logger.error(f"创建配置文件失败: {e}")
            return False
    else:
        logger.warning(f"配置文件 {config_file} 已存在")
        return False


def validate_config(config):
    """
    验证配置的完整性和有效性

    Args:
        config: 配置字典

    Returns:
        tuple: (是否有效, 错误信息列表, 警告信息列表)
    """
    errors = []
    warnings = []

    # 检查必要字段
    required_fields = [
        ("video_path", str),
        ("model_path", str),
        ("frame_save_dir", str),
        ("output_report", str),
    ]

    for field, expected_type in required_fields:
        if field not in config:
            errors.append(f"缺少必要字段: {field}")
        elif not isinstance(config[field], expected_type):
            errors.append(f"字段类型错误: {field} 应为 {expected_type.__name__}")

    # 检查数值范围
    if "yolo_confidence" in config and not (0 <= config["yolo_confidence"] <= 1):
        errors.append(f"yolo_confidence 必须在 0 到 1 之间，当前值: {config['yolo_confidence']}")

    if "yolo_iou" in config and not (0 <= config["yolo_iou"] <= 1):
        errors.append(f"yolo_iou 必须在 0 到 1 之间，当前值: {config['yolo_iou']}")

    # 检查 PaddleOCR 配置
    paddle_ocr_config = config.get("paddle_ocr", {})

    # 检查语言配置
    supported_langs = ["ch", "en", "fr", "de", "ko", "ja", "es", "pt", "ru", "ar", "hi"]
    ocr_lang = paddle_ocr_config.get("lang", "ch")
    if ocr_lang not in supported_langs:
        warnings.append(f"PaddleOCR语言 '{ocr_lang}' 可能不支持，建议使用: {', '.join(supported_langs)}")

    # 检查置信度阈值
    min_confidence = paddle_ocr_config.get("min_confidence", 0.5)
    if not (0 <= min_confidence <= 1):
        errors.append(f"paddle_ocr.min_confidence 必须在 0 到 1 之间，当前值: {min_confidence}")

    # 检查 AI 服务配置
    ai_config = config.get("ai_service", {})
    if not ai_config.get("url") or "127.0.0.1" in ai_config.get("url", ""):
        warnings.append("AI服务使用默认地址，请确保Ollama服务已启动")

    # 检查镜像检测配置
    mirror_config = config.get("mirror_detection", {})
    if mirror_config.get("enabled", False):
        if mirror_config.get("calibration_seconds", 0) <= 0:
            errors.append("mirror_detection.calibration_seconds 必须大于 0")
        if mirror_config.get("energy_deviation_threshold", 0) <= 0:
            errors.append("mirror_detection.energy_deviation_threshold 必须大于 0")
        if mirror_config.get("ocr_check_interval", 0) <= 0:
            errors.append("mirror_detection.ocr_check_interval 必须大于 0")

    return len(errors) == 0, errors, warnings