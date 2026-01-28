import os
import time
import base64
import json
import argparse
import logging
import requests
import cv2
from pathlib import Path
from ultralytics import YOLO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('defect_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 默认配置文件路径
DEFAULT_CONFIG_FILE = "config.json"

# 默认配置
DEFAULT_CONFIG = {
    "video_path": "MyVideo_1.mp4",
    "model_path": "weights/best.pt",
    "frame_save_dir": "frames",
    "output_report": "defect_report.json",
    "max_lost_frames": 30,
    "min_duration_frames": 30,
    "yolo_confidence": 0.3,
    "yolo_iou": 0.5,
    "aliyun_ocr": {
        "app_code": "YOUR_APP_CODE_HERE",
        "app_key": "YOUR_APP_KEY_HERE",
        "app_secret": "YOUR_APP_SECRET_HERE",
        "url": "https://gjbsb.market.alicloudapi.com/ocrservice/advanced"
    },
    "ai_service": {
        "url": "http://127.0.0.1:11434/api/generate",
        "model": "gemma3:4b",
        "timeout": 120
    }
}


def load_config(config_path=None):
    """加载配置，优先使用环境变量，其次使用配置文件"""
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
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}")
    else:
        logger.warning(f"配置文件 {config_file_path} 不存在，使用默认配置")

    # 环境变量覆盖（用于安全部署）
    env_mappings = {
        "ALI_APP_CODE": ("aliyun_ocr", "app_code"),
        "ALI_APP_KEY": ("aliyun_ocr", "app_key"),
        "ALI_APP_SECRET": ("aliyun_ocr", "app_secret"),
        "AI_SERVICE_URL": ("ai_service", "url"),
        "AI_MODEL": ("ai_service", "model"),
        "VIDEO_PATH": ("video_path", None),
        "MODEL_PATH": ("model_path", None)
    }

    for env_key, config_path in env_mappings.items():
        env_value = os.getenv(env_key)
        if env_value:
            if config_path[1]:  # 嵌套配置
                config[config_path[0]][config_path[1]] = env_value
            else:  # 顶级配置
                config[config_path[0]] = env_value

    return config, config_file_path


def create_config_file(config_path=None):
    """创建配置文件模板"""
    config_file = config_path or DEFAULT_CONFIG_FILE

    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)

    if not os.path.exists(config_file):
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        logger.info(f"已创建配置文件: {config_file}")
        logger.info("请编辑此文件并填入您的API密钥")
        return True
    else:
        logger.warning(f"配置文件 {config_file} 已存在")
        return False


class DefectDetector:
    def __init__(self, video_path=None, model_path=None, config_path=None):
        # 加载配置
        self.config, self.config_file_path = load_config(config_path)

        self.video_path = video_path or self.config["video_path"]
        self.model_path = model_path or self.config["model_path"]

        # 初始化模型
        logger.info(f"加载YOLO模型: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

        # 初始化视频捕获
        logger.info(f"打开视频文件: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logger.error(f"无法打开视频文件: {self.video_path}")
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_idx = 0

        # 创建保存目录
        self.frame_save_dir = Path(self.config["frame_save_dir"])
        self.frame_save_dir.mkdir(exist_ok=True)

        # 追踪相关
        self.active_tracks = {}
        self.final_results = []
        self.max_lost_frames = self.config["max_lost_frames"]
        self.min_duration_frames = self.config["min_duration_frames"]

        # OCR配置
        self.aliyun_config = self.config["aliyun_ocr"]
        self.ai_config = self.config["ai_service"]

        logger.info("缺陷检测器初始化完成")

    def save_frame_image(self, track_id, tag, frame):
        """保存帧图像"""
        filename = f"track_{track_id}_{tag}_{self.current_frame_idx}.jpg"
        path = self.frame_save_dir / filename
        cv2.imwrite(str(path), frame)
        return str(path)

    def aliyun_ocr(self, image_path):
        """调用阿里云OCR"""
        if not self.aliyun_config["app_code"] or self.aliyun_config["app_code"] == "YOUR_APP_CODE_HERE":
            logger.warning("阿里云OCR未配置或使用默认值，跳过识别")
            return ""

        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            headers = {
                "Authorization": f"APPCODE {self.aliyun_config['app_code']}",
                "Content-Type": "application/json; charset=UTF-8"
            }
            payload = {
                "img": img_b64,
                "prob": True,
                "charInfo": False,
                "rotate": True,
                "table": False
            }

            response = requests.post(
                self.aliyun_config["url"],
                headers=headers,
                data=json.dumps(payload),
                timeout=10
            )
            response.raise_for_status()

            result = response.json()
            if result.get("success"):
                lines = result.get("data", {}).get("content", [])
                return "\n".join(lines)

            # 兼容不同返回格式
            words = result.get("words", []) or result.get("prism_wordsInfo", [])
            if isinstance(words, list):
                return "\n".join(item.get("word", "") for item in words if isinstance(item, dict))

            return ""

        except requests.exceptions.RequestException as e:
            logger.error(f"OCR请求失败: {e}")
            return ""
        except Exception as e:
            logger.error(f"OCR处理异常: {e}")
            return ""

    def get_ai_summary(self, ocr_text):
        """使用本地AI整理OCR结果"""
        if not ocr_text or len(ocr_text.strip()) < 10:
            return {
                "任务名称": "未识别",
                "检测地点": "未识别",
                "井号区间": "未识别",
                "管材": "未识别",
                "管径": "未识别",
                "当前距离": "未识别"
            }

        # 提取关键文本（只取前500字符）
        short_text = ocr_text[:500]

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
            "model": self.ai_config["model"],
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post(
                self.ai_config["url"],
                json=payload,
                timeout=self.ai_config["timeout"]
            )
            response.raise_for_status()

            result_text = response.json().get("response", "")

            # 清理JSON响应
            result_text = result_text.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            # 解析JSON
            if result_text.startswith("{") and result_text.endswith("}"):
                return json.loads(result_text)
            else:
                # 尝试提取JSON
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return {"error": "AI返回格式无效"}

        except requests.exceptions.ConnectionError:
            logger.warning("无法连接到AI服务，请确保Ollama已启动")
            return {"error": "AI服务未连接"}
        except Exception as e:
            logger.error(f"AI处理失败: {e}")
            return {"error": str(e)}

    def run_ocr_for_track(self, track):
        """对缺陷进行OCR和AI分析"""
        texts = []

        # 只分析关键帧（开始帧和最佳帧）
        for key in ("start_frame_path", "best_frame_path"):
            path = track.get(key)
            if path and os.path.exists(path):
                try:
                    text = self.aliyun_ocr(path)
                    if text:
                        texts.append(f"[{key}] {text}")
                except Exception as e:
                    logger.warning(f"{key} OCR失败: {e}")

        ocr_text = "\n".join(texts) if texts else "无OCR文本"
        track["ocr_text"] = ocr_text

        # 获取AI总结
        if texts:
            track["ai_summary"] = self.get_ai_summary(ocr_text)
        else:
            track["ai_summary"] = {"error": "无OCR文本可分析"}

    def process_frame(self, frame):
        """处理单帧"""
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.config["yolo_confidence"],
            iou=self.config["yolo_iou"]
        )

        present_ids = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            cls_ids = results[0].boxes.cls.int().cpu().numpy()
            names = self.model.names

            for box, conf, tid, cid in zip(boxes, confidences, track_ids, cls_ids):
                present_ids.append(tid)

                if tid not in self.active_tracks:
                    # 新缺陷
                    self.active_tracks[tid] = {
                        "track_id": int(tid),
                        "defect_type": names[cid],
                        "start_frame": self.current_frame_idx,
                        "end_frame": self.current_frame_idx,
                        "start_time": self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                        "end_time": self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                        "max_conf": float(conf),
                        "last_seen": self.current_frame_idx,
                        "start_frame_path": self.save_frame_image(tid, "start", frame),
                        "best_frame_path": None
                    }
                else:
                    # 更新现有缺陷
                    track = self.active_tracks[tid]
                    track["end_frame"] = self.current_frame_idx
                    track["end_time"] = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    track["last_seen"] = self.current_frame_idx

                    # 更新最佳置信度帧
                    if conf > track["max_conf"]:
                        track["max_conf"] = float(conf)
                        if track.get("best_frame_path") and os.path.exists(track["best_frame_path"]):
                            try:
                                os.remove(track["best_frame_path"])
                            except:
                                pass
                        track["best_frame_path"] = self.save_frame_image(tid, "best", frame)

                logger.info(f"缺陷: {names[cid]}, ID:{tid}, 置信度: {conf:.3f}, 帧: {self.current_frame_idx}")

        # 结算已丢失的ID
        ids_to_settle = []
        for tid, info in self.active_tracks.items():
            if self.current_frame_idx - info["last_seen"] > self.max_lost_frames:
                ids_to_settle.append(tid)

        for tid in ids_to_settle:
            self.settle_track(tid)

        return results[0].plot() if results else frame

    def settle_track(self, track_id):
        """结算一个缺陷轨迹"""
        if track_id not in self.active_tracks:
            return

        track = self.active_tracks.pop(track_id)
        duration_frames = track["end_frame"] - track["start_frame"] + 1

        if duration_frames >= self.min_duration_frames:
            logger.info(f"✅ ID:{track_id} 持续{duration_frames}帧，进行OCR/AI分析")

            # 如果没有最佳帧，使用开始帧
            if not track.get("best_frame_path"):
                track["best_frame_path"] = track["start_frame_path"]

            # 删除辅助字段
            track.pop("last_seen", None)

            # 进行OCR和AI分析
            try:
                self.run_ocr_for_track(track)
            except Exception as e:
                logger.error(f"ID:{track_id} OCR/AI分析失败: {e}")
                track["ocr_text"] = f"处理失败: {e}"
                track["ai_summary"] = {"error": str(e)}

            self.final_results.append(track)
        else:
            logger.info(f"⏩ ID:{track_id} 仅持续{duration_frames}帧，跳过分析")

            # 清理图片文件
            for key in ["start_frame_path", "best_frame_path"]:
                path = track.get(key)
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
                track.pop(key, None)

            track["ocr_text"] = f"持续时间不足{self.min_duration_frames}帧，跳过分析"
            track["ai_summary"] = {"note": "持续时间过短，未进行分析"}
            track.pop("last_seen", None)
            self.final_results.append(track)

    def generate_report(self, output_path=None):
        """生成最终报告"""
        # 结算剩余缺陷
        for track_id in list(self.active_tracks.keys()):
            self.settle_track(track_id)

        # 按开始帧排序
        self.final_results.sort(key=lambda x: x["start_frame"])

        # 准备报告数据
        report_data = {
            "video_info": {
                "path": self.video_path,
                "fps": self.fps,
                "total_frames": self.current_frame_idx,
                "total_duration": self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            },
            "detection_config": {
                "model": self.model_path,
                "confidence_threshold": self.config["yolo_confidence"],
                "iou_threshold": self.config["yolo_iou"],
                "max_lost_frames": self.max_lost_frames,
                "min_duration_frames": self.min_duration_frames
            },
            "defects": self.final_results,
            "summary": {
                "total_defects": len(self.final_results),
                "analyzed_defects": sum(1 for d in self.final_results
                                        if "持续时间不足" not in d.get("ocr_text", "")),
                "defect_types": {}
            }
        }

        # 统计缺陷类型
        for defect in self.final_results:
            defect_type = defect["defect_type"]
            report_data["summary"]["defect_types"][defect_type] = \
                report_data["summary"]["defect_types"].get(defect_type, 0) + 1

        # 保存报告
        output_path = output_path or self.config["output_report"]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"报告已保存到: {output_path}")
        return report_data

    def run(self, show_video=False):
        """运行检测"""
        logger.info("开始缺陷检测...")

        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break

                # 处理当前帧
                annotated_frame = self.process_frame(frame)

                # 显示视频（可选）
                if show_video:
                    cv2.imshow('Defect Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("用户手动停止")
                        break

                self.current_frame_idx += 1

                # 每100帧输出进度
                if self.current_frame_idx % 100 == 0:
                    logger.info(f"已处理 {self.current_frame_idx} 帧，当前活动缺陷: {len(self.active_tracks)}个")

        except KeyboardInterrupt:
            logger.info("检测被用户中断")
        except Exception as e:
            logger.error(f"检测过程中出现错误: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

            # 生成报告
            report = self.generate_report()

            # 打印总结
            self.print_summary(report)

            logger.info("缺陷检测完成")

    def print_summary(self, report):
        """打印检测总结"""
        summary = report["summary"]

        print("\n" + "=" * 60)
        print("缺陷检测总结报告")
        print("=" * 60)
        print(f"视频文件: {report['video_info']['path']}")
        print(f"总帧数: {report['video_info']['total_frames']}")
        print(f"持续时间: {report['video_info']['total_duration']:.2f}秒")
        print(f"帧率: {report['video_info']['fps']:.1f} FPS")
        print(f"\n检测到缺陷总数: {summary['total_defects']}个")
        print(f"详细分析缺陷: {summary['analyzed_defects']}个")
        print(f"跳过分析缺陷: {summary['total_defects'] - summary['analyzed_defects']}个")

        if summary['defect_types']:
            print("\n缺陷类型统计:")
            for defect_type, count in summary['defect_types'].items():
                print(f"  {defect_type}: {count}个")

        print("\n" + "=" * 60)
        print("详细缺陷列表")
        print("=" * 60)

        for i, defect in enumerate(report['defects'], 1):
            duration_frames = defect['end_frame'] - defect['start_frame'] + 1
            duration_time = defect['end_time'] - defect['start_time']

            print(f"\n{i}. ID:{defect.get('track_id', 'N/A')} - {defect['defect_type']}")
            print(f"   帧范围: {defect['start_frame']}-{defect['end_frame']} (共{duration_frames}帧)")
            print(f"   时间: {defect['start_time']:.1f}s - {defect['end_time']:.1f}s (共{duration_time:.1f}s)")
            print(f"   最高置信度: {defect['max_conf']:.4f}")

            if "ai_summary" in defect:
                summary = defect["ai_summary"]
                if isinstance(summary, dict) and "error" not in summary:
                    print(f"   AI提取信息:")
                    for key, value in summary.items():
                        if value and value not in ["未识别", ""]:
                            print(f"     {key}: {value}")
                elif isinstance(summary, dict) and "error" in summary:
                    print(f"   AI分析错误: {summary['error']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='管道缺陷智能检测系统')
    parser.add_argument('--video', type=str, help='视频文件路径')
    parser.add_argument('--model', type=str, help='YOLO模型路径')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_FILE,
                        help=f'配置文件路径 (默认: {DEFAULT_CONFIG_FILE})')
    parser.add_argument('--show', action='store_true', help='显示检测视频窗口')
    parser.add_argument('--create-config', action='store_true', help='创建配置文件模板')

    args = parser.parse_args()

    # 创建配置文件模板
    if args.create_config:
        create_config_file(args.config)
        return

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        logger.warning(f"配置文件 {args.config} 不存在")
        create = input("是否创建默认配置文件? (y/n): ")
        if create.lower() == 'y':
            create_config_file(args.config)
        else:
            logger.info("使用默认配置继续运行...")

    try:
        # 初始化检测器
        detector = DefectDetector(
            video_path=args.video,
            model_path=args.model,
            config_path=args.config
        )

        # 运行检测
        detector.run(show_video=args.show)

    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()