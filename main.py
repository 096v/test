#main.py
import os
import sys
import time
import argparse
import logging
import json
import cv2
from pathlib import Path
from ultralytics import YOLO

# 导入自定义模块
from config_loader import load_config, create_config_file, validate_config
from mirror_manager import MirrorManager
from utils import get_ai_summary_api, save_frame_with_metadata, NumpyEncoder, paddle_ocr_api


# 创建自定义的StreamHandler以处理Windows编码问题
class SafeStreamHandler(logging.StreamHandler):
    """安全的StreamHandler，处理Windows控制台的编码问题"""

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream

            # 尝试使用utf-8编码，如果失败则使用replace策略
            try:
                if hasattr(stream, 'encoding') and stream.encoding:
                    stream.write(msg + self.terminator)
                else:
                    # Windows控制台可能没有encoding属性
                    stream.write(
                        (msg + self.terminator).encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
            except UnicodeEncodeError:
                # 如果还是失败，使用replace策略
                safe_msg = (msg + self.terminator).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                stream.write(safe_msg)
            self.flush()
        except Exception:
            self.handleError(record)


# 配置日志
def setup_logging():
    """配置日志系统"""
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除现有的handler
    if logger.handlers:
        logger.handlers.clear()

    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件handler - 使用UTF-8编码
    file_handler = logging.FileHandler('defect_detection.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台handler - 使用安全的handler
    console_handler = SafeStreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# 设置日志
logger = setup_logging()


# main.py (部分修改)
# ... 前面的导入和类定义保持不变 ...

class DefectDetector:
    """管道缺陷智能检测器"""

    def __init__(self, video_path=None, model_path=None, config_path=None, no_mirror=False):
        """
        初始化缺陷检测器

        Args:
            video_path: 视频文件路径
            model_path: 模型文件路径
            config_path: 配置文件路径
            no_mirror: 是否禁用镜像检测
        """
        # 加载配置
        self.config, self.config_file_path = load_config(config_path)

        # 配置验证
        is_valid, errors, warnings = validate_config(self.config)
        if not is_valid:
            logger.error("配置验证失败:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError("配置验证失败，请检查配置文件")

        for warning in warnings:
            logger.warning(f"配置警告: {warning}")

        # 覆盖配置
        self.video_path = video_path or self.config["video_path"]
        self.model_path = model_path or self.config["model_path"]

        # 加载YOLO模型
        logger.info(f"加载YOLO模型: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"模型加载成功，类别: {list(self.model.names.values())}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

        # 打开视频文件
        logger.info(f"打开视频文件: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logger.error(f"无法打开视频文件: {self.video_path}")
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        # 获取视频信息
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"视频信息: {self.frame_width}x{self.frame_height}, {self.fps:.1f}FPS, 总帧数: {self.total_frames}")

        self.current_frame_idx = 0

        # 创建保存目录
        self.frame_save_dir = Path(self.config["frame_save_dir"])
        self.frame_save_dir.mkdir(exist_ok=True)

        # 初始化镜像管理器
        self.mirror_enabled = self.config.get("mirror_detection", {}).get("enabled", True) and not no_mirror
        if self.mirror_enabled:
            logger.info("初始化镜像自动纠正系统...")

            # 创建OCR函数闭包 - 修改为使用PaddleOCR
            def ocr_wrapper(image_path):
                return paddle_ocr_api(image_path, self.config["paddle_ocr"])

            self.mirror_manager = MirrorManager(
                ocr_func=ocr_wrapper,
                fps=self.fps,
                config=self.config.get("mirror_detection", {})
            )
        else:
            logger.info("镜像自动纠正已禁用")
            self.mirror_manager = None

        # 追踪相关
        self.active_tracks = {}
        self.final_results = []
        self.max_lost_frames = self.config["max_lost_frames"]
        self.min_duration_frames = self.config["min_duration_frames"]

        logger.info("缺陷检测器初始化完成")
    def save_frame_image(self, track_id, tag, frame):
        """保存帧图像"""
        filename = f"track_{track_id}_{tag}_{self.current_frame_idx}.jpg"
        path = self.frame_save_dir / filename

        # 添加元数据
        metadata = {
            "track_id": track_id,
            "tag": tag,
            "frame_idx": self.current_frame_idx,
            "timestamp": time.time(),
            "mirror_status": self.mirror_manager.get_status() if self.mirror_manager else None
        }

        return save_frame_with_metadata(frame, self.frame_save_dir, filename, metadata)

    def run_ocr_for_track(self, track):
        """对缺陷进行OCR和AI分析"""
        texts = []

        # 分析关键帧
        for key in ("start_frame_path", "best_frame_path"):
            path = track.get(key)
            if path and os.path.exists(path):
                try:
                    # 修改为使用PaddleOCR
                    text = paddle_ocr_api(path, self.config["paddle_ocr"])
                    if text:
                        texts.append(f"[{key}] {text}")
                except Exception as e:
                    logger.warning(f"{key} OCR失败: {e}")

        ocr_text = "\n".join(texts) if texts else "无OCR文本"
        track["ocr_text"] = ocr_text

        # 获取AI总结
        if texts:
            track["ai_summary"] = get_ai_summary_api(ocr_text, self.config["ai_service"])
        else:
            track["ai_summary"] = {"error": "无OCR文本可分析"}
    def process_frame(self, frame):
        """处理单帧"""
        # 应用镜像纠正
        if self.mirror_manager:
            frame = self.mirror_manager.update_and_correct(frame, self.current_frame_idx)

        # YOLO检测与追踪
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.config["yolo_confidence"],
            iou=self.config["yolo_iou"],
            verbose=False
        )

        present_ids = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
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

                logger.debug(f"缺陷: {names[cid]}, ID:{tid}, 置信度: {conf:.3f}, 帧: {self.current_frame_idx}")

        # 结算已丢失的ID
        ids_to_settle = []
        for tid, info in self.active_tracks.items():
            if self.current_frame_idx - info["last_seen"] > self.max_lost_frames:
                ids_to_settle.append(tid)

        for tid in ids_to_settle:
            self.settle_track(tid)

        # 绘制检测结果
        if results:
            annotated_frame = results[0].plot()

            # 添加镜像状态显示
            if self.mirror_manager and self.mirror_manager.is_calibrated:
                status_text = "镜像模式" if self.mirror_manager.is_mirrored else "正常模式"
                color = (0, 0, 255) if self.mirror_manager.is_mirrored else (0, 255, 0)
                cv2.putText(annotated_frame, f"状态: {status_text}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 添加帧信息
            cv2.putText(annotated_frame, f"帧: {self.current_frame_idx}/{self.total_frames}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"活动缺陷: {len(self.active_tracks)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return annotated_frame

        return frame

    def settle_track(self, track_id):
        """结算一个缺陷轨迹"""
        if track_id not in self.active_tracks:
            return

        track = self.active_tracks.pop(track_id)
        duration_frames = track["end_frame"] - track["start_frame"] + 1

        if duration_frames >= self.min_duration_frames:
            # 使用ASCII字符代替Unicode字符，避免Windows编码问题
            logger.info(f"[分析] ID:{track_id} 持续{duration_frames}帧，进行OCR/AI分析")

            # 如果没有最佳帧，使用开始帧
            if not track.get("best_frame_path"):
                track["best_frame_path"] = track["start_frame_path"]

            # 删除辅助字段
            track.pop("last_seen", None)

            # 记录镜像状态
            if self.mirror_manager:
                track["mirror_correction"] = self.mirror_manager.get_status()

            # 进行OCR和AI分析
            try:
                self.run_ocr_for_track(track)
            except Exception as e:
                logger.error(f"ID:{track_id} OCR/AI分析失败: {e}")
                track["ocr_text"] = f"处理失败: {e}"
                track["ai_summary"] = {"error": str(e)}

            self.final_results.append(track)
        else:
            # 使用ASCII字符代替Unicode字符，避免Windows编码问题
            logger.info(f"[跳过] ID:{track_id} 仅持续{duration_frames}帧，跳过分析")

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

        # 镜像管理器状态
        mirror_status = self.mirror_manager.get_status() if self.mirror_manager else {
            'enabled': False,
            'is_mirrored': False
        }

        # 准备报告数据
        report_data = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_info": {
                "path": self.video_path,
                "resolution": f"{self.frame_width}x{self.frame_height}",
                "fps": self.fps,
                "total_frames": self.current_frame_idx,
                "total_duration": self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if self.cap else 0
            },
            "detection_config": {
                "model": self.model_path,
                "confidence_threshold": self.config["yolo_confidence"],
                "iou_threshold": self.config["yolo_iou"],
                "max_lost_frames": self.max_lost_frames,
                "min_duration_frames": self.min_duration_frames
            },
            "mirror_correction": {
                "enabled": self.mirror_enabled,
                "status": mirror_status
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
            json.dump(report_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        logger.info(f"报告已保存到: {output_path}")
        return report_data

    def print_summary(self, report):
        """打印检测总结"""
        summary = report["summary"]
        mirror_info = report.get("mirror_correction", {})

        print("\n" + "=" * 60)
        print("缺陷检测总结报告")
        print("=" * 60)
        print(f"视频文件: {report['video_info']['path']}")
        print(f"分辨率: {report['video_info']['resolution']}")
        print(f"总帧数: {report['video_info']['total_frames']}")
        print(f"持续时间: {report['video_info']['total_duration']:.2f}秒")
        print(f"帧率: {report['video_info']['fps']:.1f} FPS")

        # 镜像纠正信息
        if mirror_info.get('enabled'):
            mirror_status = mirror_info.get('status', {})
            print(f"\n镜像自动纠正: 启用")
            print(f"最终状态: {'镜像模式' if mirror_status.get('is_mirrored') else '正常模式'}")
            if 'stats' in mirror_status:
                stats = mirror_status['stats']
                print(f"镜像检测统计: {stats.get('total_checks', 0)}次检查, "
                      f"{stats.get('mirror_flips', 0)}次翻转, "
                      f"{stats.get('ocr_triggered', 0)}次OCR验证")

        print(f"\n检测到缺陷总数: {summary['total_defects']}个")
        print(f"详细分析缺陷: {summary['analyzed_defects']}个")
        print(f"跳过分析缺陷: {summary['total_defects'] - summary['analyzed_defects']}个")

        if summary['defect_types']:
            print("\n缺陷类型统计:")
            for defect_type, count in sorted(summary['defect_types'].items()):
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

            # 显示镜像信息
            if 'mirror_correction' in defect:
                mc = defect['mirror_correction']
                if mc.get('is_mirrored'):
                    print(f"   镜像纠正: 已应用")

            if "ai_summary" in defect:
                summary = defect["ai_summary"]
                if isinstance(summary, dict) and "error" not in summary:
                    print(f"   AI提取信息:")
                    for key, value in summary.items():
                        if value and value not in ["未识别", ""]:
                            print(f"     {key}: {value}")
                elif isinstance(summary, dict) and "error" in summary:
                    print(f"   AI分析错误: {summary['error']}")

    def run(self, show_video=False):
        """运行检测"""
        logger.info("开始缺陷检测...")

        start_time = time.time()

        if self.mirror_manager and self.mirror_enabled:
            logger.info(f"镜像自动纠正已启用，学习阶段: 前{self.mirror_manager.calibration_frames}帧")

        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break

                # 处理当前帧
                annotated_frame = self.process_frame(frame)

                # 显示视频（可选）
                if show_video:
                    cv2.imshow('管道缺陷检测系统', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("用户手动停止")
                        break

                self.current_frame_idx += 1

                # 进度输出
                if self.current_frame_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_processed = self.current_frame_idx / elapsed
                    remaining_frames = max(0, self.total_frames - self.current_frame_idx)
                    estimated_time = remaining_frames / fps_processed if fps_processed > 0 else 0

                    logger.info(f"进度: {self.current_frame_idx}/{self.total_frames} 帧 "
                                f"({self.current_frame_idx / self.total_frames * 100:.1f}%), "
                                f"处理速度: {fps_processed:.1f} FPS, "
                                f"预计剩余时间: {estimated_time:.1f}秒, "
                                f"活动缺陷: {len(self.active_tracks)}个")

                    # 镜像统计信息
                    if self.mirror_manager and self.mirror_manager.is_calibrated:
                        stats = self.mirror_manager.stats
                        logger.debug(f"镜像检查: {stats['total_checks']}次, 翻转: {stats['mirror_flips']}次")

        except KeyboardInterrupt:
            logger.info("检测被用户中断")
        except Exception as e:
            logger.error(f"检测过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

            # 计算总耗时
            total_time = time.time() - start_time
            logger.info(f"处理完成，总耗时: {total_time:.1f}秒，平均速度: {self.current_frame_idx / total_time:.1f} FPS")

            # 生成报告
            report = self.generate_report()

            # 打印总结
            self.print_summary(report)

            logger.info("缺陷检测完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='管道缺陷智能检测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python main.py --video test.mp4

  # 显示检测窗口
  python main.py --video test.mp4 --show

  # 禁用镜像检测
  python main.py --video test.mp4 --no-mirror

  # 创建配置文件
  python main.py --create-config

  # 使用自定义配置
  python main.py --config custom_config.json
        """
    )

    parser.add_argument('--video', type=str, help='视频文件路径')
    parser.add_argument('--model', type=str, help='YOLO模型路径')
    parser.add_argument('--config', type=str, default="config.json",
                        help='配置文件路径 (默认: config.json)')
    parser.add_argument('--show', action='store_true', help='显示检测视频窗口')
    parser.add_argument('--create-config', action='store_true', help='创建配置文件模板')
    parser.add_argument('--no-mirror', action='store_true', help='禁用镜像自动纠正')
    parser.add_argument('--verbose', '-v', action='store_true', help='启用详细日志输出')
    parser.add_argument('--ascii-only', action='store_true', help='仅使用ASCII字符输出，避免编码问题')

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("启用详细日志输出")

    # 创建配置文件模板
    if args.create_config:
        if create_config_file(args.config):
            print(f"配置文件已创建: {args.config}")
            print("请编辑此文件并填入您的API密钥和配置参数")
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
            config_path=args.config,
            no_mirror=args.no_mirror
        )

        # 运行检测
        detector.run(show_video=args.show)

    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()