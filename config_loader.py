import json
import logging
import os

logger = logging.getLogger(__name__)

# 默认配置 - 添加OCR提取策略配置
DEFAULT_CONFIG = {
    "video_path": "MyVideo_1.mp4",
    "model_path": "weights/best.pt",
    "frame_save_dir": "frames",
    "output_report": "defect_report.json",
    "max_lost_frames": 30,
    "min_duration_frames": 30,
    "yolo_confidence": 0.3,
    "yolo_iou": 0.5,

    "base64_encoding": {
        "quality": 80,
        "max_width": 800
    },

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
        "timeout": 120,
        "max_retries": 3
    },

    "ocr_extraction": {
        "enabled": True,
        "strategy": "best_frame_sampling",  # 策略：best_frame_sampling | multi_frame_merge | first_n_frames
        "sample_frames": 30,  # 采样帧数
        "min_score_threshold": 0.3,  # 最低评分阈值
        "preprocessing": {
            "enabled": True,
            "adaptive_threshold": True,
            "histogram_equalization": True,
            "denoise": True
        },
        "scoring": {
            "keyword_weight": 0.45,
            "digit_weight": 0.20,
            "structure_weight": 0.15,
            "length_weight": 0.15,
            "valid_ratio_weight": 0.05
        },
        "keywords": [
            "井", "雨", "污", "管", "道", "检测",
            "S", "Q", "Φ", "DN", "编号", "人员",
            "雨水", "污水", "给水", "管道", "井号",
            "材质", "方向", "检测井", "检查井"
        ],
        "debug_mode": False,
        "save_debug_frames": False,
        "debug_dir": "debug/ocr_best_frames"
    },

    "tracking_config": {
        "tracker": "bytetrack.yaml",
        "persist": True
    },

    "logging": {
        "level": "INFO",
        "file": "defect_detection.log",
        "max_file_size_mb": 10,
        "backup_count": 3
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
                # 深度合并配置（避免覆盖嵌套结构）
                deep_merge(config, file_config)
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
        "AI_MAX_RETRIES": ("ai_service", "max_retries"),

        # OCR提取配置
        "OCR_EXTRACTION_ENABLED": ("ocr_extraction", "enabled"),
        "OCR_SAMPLE_FRAMES": ("ocr_extraction", "sample_frames"),
        "OCR_MIN_SCORE_THRESHOLD": ("ocr_extraction", "min_score_threshold"),
        "OCR_DEBUG_MODE": ("ocr_extraction", "debug_mode"),
        "OCR_SAVE_DEBUG_FRAMES": ("ocr_extraction", "save_debug_frames"),

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
        "MIN_DURATION_FRAMES": ("min_duration_frames", None),
        "BASE64_QUALITY": ("base64_encoding", "quality"),
        "BASE64_MAX_WIDTH": ("base64_encoding", "max_width")
    }

    for env_key, config_path_tuple in env_mappings.items():
        env_value = os.getenv(env_key)
        if env_value is not None:
            section, key = config_path_tuple
            if key:  # 嵌套配置
                # 处理布尔值
                bool_keys = [
                    "PADDLE_OCR_USE_ANGLE_CLS", "PADDLE_OCR_USE_GPU",
                    "PADDLE_OCR_SHOW_LOG", "MIRROR_DETECTION_ENABLED",
                    "OCR_EXTRACTION_ENABLED", "OCR_DEBUG_MODE", "OCR_SAVE_DEBUG_FRAMES"
                ]
                if env_key in bool_keys:
                    config[section][key] = env_value.lower() in ('true', '1', 'yes', 't')
                # 处理数值类型
                elif env_key in [
                    "PADDLE_OCR_GPU_MEM", "PADDLE_OCR_CPU_THREADS",
                    "PADDLE_OCR_DET_DB_THRESH", "PADDLE_OCR_DET_DB_BOX_THRESH",
                    "PADDLE_OCR_DROP_SCORE", "PADDLE_OCR_MIN_CONFIDENCE",
                    "MIRROR_CALIBRATION_SECONDS", "MIRROR_DEVIATION_THRESHOLD",
                    "MIRROR_CHECK_INTERVAL", "YOLO_CONFIDENCE", "YOLO_IOU",
                    "MAX_LOST_FRAMES", "MIN_DURATION_FRAMES", "AI_TIMEOUT",
                    "AI_MAX_RETRIES", "OCR_SAMPLE_FRAMES", "OCR_MIN_SCORE_THRESHOLD",
                    "BASE64_QUALITY", "BASE64_MAX_WIDTH"
                ]:
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


def deep_merge(target, source):
    """
    深度合并两个字典

    Args:
        target: 目标字典
        source: 源字典
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_merge(target[key], value)
        else:
            target[key] = value


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
            print(f"\n✅ 配置文件已创建: {config_file}")
            print("请编辑以下重要配置：")
            print("  - video_path: 视频文件路径")
            print("  - model_path: YOLO模型路径")
            print("  - ai_service.url: AI服务地址")
            print("  - ai_service.model: AI模型名称")
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
    if ai_config.get("timeout", 0) < 30:
        warnings.append("AI服务超时时间设置较短，建议设置为60秒以上")

    # 检查镜像检测配置
    mirror_config = config.get("mirror_detection", {})
    if mirror_config.get("enabled", False):
        if mirror_config.get("calibration_seconds", 0) <= 0:
            errors.append("mirror_detection.calibration_seconds 必须大于 0")
        if mirror_config.get("energy_deviation_threshold", 0) <= 0:
            errors.append("mirror_detection.energy_deviation_threshold 必须大于 0")
        if mirror_config.get("ocr_check_interval", 0) <= 0:
            errors.append("mirror_detection.ocr_check_interval 必须大于 0")

    # 检查 OCR 提取配置
    ocr_extraction_config = config.get("ocr_extraction", {})
    if ocr_extraction_config.get("enabled", True):
        sample_frames = ocr_extraction_config.get("sample_frames", 30)
        if sample_frames <= 0 or sample_frames > 100:
            warnings.append(f"ocr_extraction.sample_frames 值 {sample_frames} 可能不合理，建议在1-100之间")

        min_score = ocr_extraction_config.get("min_score_threshold", 0.3)
        if not (0 <= min_score <= 1):
            errors.append(f"ocr_extraction.min_score_threshold 必须在 0 到 1 之间，当前值: {min_score}")

        # 检查评分权重总和
        scoring = ocr_extraction_config.get("scoring", {})
        weight_sum = sum(scoring.values())
        if abs(weight_sum - 1.0) > 0.01:
            warnings.append(f"OCR评分权重总和应为1.0，当前为 {weight_sum:.2f}")

    # 检查 Base64 配置
    base64_config = config.get("base64_encoding", {})
    quality = base64_config.get("quality", 80)
    if not (1 <= quality <= 100):
        warnings.append(f"base64_encoding.quality 应在 1-100 之间，当前值: {quality}")

    max_width = base64_config.get("max_width", 800)
    if max_width <= 0:
        warnings.append(f"base64_encoding.max_width 应大于 0，当前值: {max_width}")

    # 输出验证结果
    if errors:
        logger.error(f"配置验证发现 {len(errors)} 个错误: {errors}")
    if warnings:
        logger.warning(f"配置验证发现 {len(warnings)} 个警告: {warnings}")

    return len(errors) == 0, errors, warnings


def save_config(config, config_path=None):
    """
    保存配置到文件

    Args:
        config: 配置字典
        config_path: 配置文件路径

    Returns:
        bool: 是否保存成功
    """
    config_file = config_path or DEFAULT_CONFIG_FILE

    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"配置已保存到: {config_file}")
        return True
    except Exception as e:
        logger.error(f"保存配置失败: {e}")
        return False


def print_config_summary(config):
    """
    打印配置摘要

    Args:
        config: 配置字典
    """
    print("\n" + "=" * 60)
    print("配置摘要")
    print("=" * 60)

    print(f" 视频文件: {config.get('video_path', '未设置')}")
    print(f" 模型文件: {config.get('model_path', '未设置')}")
    print(f" 输出报告: {config.get('output_report', '未设置')}")

    print("\n 检测配置:")
    print(f"  - YOLO置信度: {config.get('yolo_confidence', 0.3)}")
    print(f"  - YOLO IoU: {config.get('yolo_iou', 0.5)}")
    print(f"  - 最小持续时间帧: {config.get('min_duration_frames', 30)}")
    print(f"  - 最大丢失帧: {config.get('max_lost_frames', 30)}")

    print("\n OCR提取配置:")
    ocr_extraction = config.get('ocr_extraction', {})
    print(f"  - 是否启用: {ocr_extraction.get('enabled', True)}")
    print(f"  - 采样帧数: {ocr_extraction.get('sample_frames', 30)}")
    print(f"  - 最低评分阈值: {ocr_extraction.get('min_score_threshold', 0.3)}")

    print("\n AI服务配置:")
    ai_service = config.get('ai_service', {})
    print(f"  - URL: {ai_service.get('url', '未设置')}")
    print(f"  - 模型: {ai_service.get('model', '未设置')}")

    print("=" * 60)