# -*- coding: utf-8 -*-

"""
YOLO-based human detection with dwell-time alert for parking enforcement staff.

このスクリプトは以下の要件を満たす「駐車監視員の可能性が高い対象」を検知し、
指定した監視エリア（自車周囲）に **5秒以上連続滞在** した場合にアラームのみを
1回鳴動させます。

要件に沿った人物特徴（YOLO-World のテキストプロンプトで指定）:
    - 緑色または黄緑色の制服
    - 蛍光色の反射安全ベスト
    - 制帽または作業用キャップ
    - 肩掛けの業務用バッグ
    - 手にハンディ端末・スマートフォン・撮影機器
    - 路上に駐車された車両のフロント付近またはナンバープレート周辺を確認する動作

動作概要:
    - YOLO (デフォルト ultralytics/yolov8n) で動画ストリームを推論。
    - 監視エリアに対象人物が入り続けた秒数を計測し、5秒経過直後にベルを鳴らす。
    - 警告や通信は行わず「アラームのみ」を実行。
    - YOLO-World 重みとテキストプロンプトを指定した場合はゼロショット検出で上記特徴を判定。
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# 依存パッケージのチェック
try:
    import cv2
    from ultralytics import YOLO
except ImportError as e:
    print(f"エラー: 必要なパッケージがインストールされていません: {e}")
    print()
    print("以下のコマンドでインストールしてください:")
    print("  pip install -r requirements.txt")
    print()
    print("または、自動インストールスクリプトを実行:")
    print("  ./install_dependencies.sh")
    sys.exit(1)

# YOLO "person" class index for the COCO dataset
PERSON_CLASS_ID = 0


@dataclass
class AlertZone:
    """Axis-aligned rectangular zone in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def draw(self, frame) -> None:
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "Alert zone",
            (self.x1 + 5, self.y1 - 10 if self.y1 > 20 else self.y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def contains(self, point: Tuple[int, int]) -> bool:
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))


class ParkingAlert:
    def __init__(
        self,
        model_name: str,
        zone: AlertZone,
        score_threshold: float = 0.45,
        alert_cooldown: float = 2.0,
        text_prompts: Optional[List[str]] = None,
        imgsz: int = 640,
        alert_message: str = "⚠️ 駐車監視員を検知しました！",
        dwell_seconds: float = 5.0,
    ) -> None:
        self.model = YOLO(model_name)
        self.zone = zone
        self.score_threshold = score_threshold
        self.alert_cooldown = alert_cooldown
        self._last_alert_at = 0.0
        self.text_prompts = text_prompts
        self.imgsz = imgsz
        self.alert_message = alert_message
        # 連続滞在を測るための状態
        self._inside_since: Optional[float] = None
        self._alerted_in_current_stay = False
        self.dwell_seconds = dwell_seconds

    def _trigger_alert(self) -> None:
        now = time.time()
        if now - self._last_alert_at < self.alert_cooldown:
            return
        # Terminal bell is cross-platform and avoids extra dependencies
        sys.stdout.write("\a")
        sys.stdout.flush()
        print(self.alert_message, flush=True)
        self._last_alert_at = now
        self._alerted_in_current_stay = True

    def _parse_detections(self, results) -> Iterable[Detection]:
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                if conf < self.score_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # トレーニング済みモデルの場合、クラス名を取得
                if hasattr(result, "names") and result.names is not None:
                    label = result.names.get(cls_id, f"class_{cls_id}")
                    # 駐車監視員クラス（parking_officer）のみを検出
                    if label != "parking_officer" and cls_id != PERSON_CLASS_ID:
                        continue
                elif self.text_prompts is None:
                    # デフォルトのpersonクラスのみ
                    if cls_id != PERSON_CLASS_ID:
                        continue
                    label = "person"
                else:
                    # YOLO-World maps cls IDs to the supplied prompts via result.names
                    label = (
                        result.names.get(cls_id, f"prompt_{cls_id}")
                        if hasattr(result, "names") and result.names is not None
                        else f"prompt_{cls_id}"
                    )
                yield Detection((x1, y1, x2, y2), conf, label)

    def process_frame(self, frame):
        if self.text_prompts:
            results = self.model.predict(
                source=frame,
                text=self.text_prompts,
                conf=self.score_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )
        else:
            results = self.model(frame, imgsz=self.imgsz, verbose=False)

        person_detections = list(self._parse_detections(results))
        in_zone = False

        for det in person_detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, det.center, 4, (255, 0, 0), -1)
            cv2.putText(
                frame,
                f"{det.label} {det.confidence:.2f}",
                (x1, y1 - 10 if y1 > 20 else y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if self.zone.contains(det.center):
                in_zone = True
                cv2.putText(
                    frame,
                    "IN ALERT ZONE",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        self.zone.draw(frame)

        now = time.time()
        if in_zone:
            if self._inside_since is None:
                self._inside_since = now
                self._alerted_in_current_stay = False
            dwell_time = now - self._inside_since
            cv2.putText(
                frame,
                f"滞在: {dwell_time:.1f}s / 目標 {self.dwell_seconds:.0f}s",
                (self.zone.x1 + 5, self.zone.y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if dwell_time >= self.dwell_seconds else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            if dwell_time >= self.dwell_seconds and not self._alerted_in_current_stay:
                self._trigger_alert()
        else:
            # 監視エリアから出たら連続滞在計測をリセット
            self._inside_since = None
            self._alerted_in_current_stay = False

        return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO-based parking enforcement alert")
    parser.add_argument(
        "--source",
        default=0,
        help="Camera index or video file path (default: 0)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model checkpoint (default: yolov8n.pt). トレーニング済みモデルを使用する場合は models/parking_officer_iterXXX.pt を指定",
    )
    parser.add_argument(
        "--zone",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Alert zone rectangle coordinates (pixels)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Minimum confidence for detections",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Seconds to wait between alert sounds",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (square)",
    )
    parser.add_argument(
        "--use-prompts",
        action="store_true",
        help="Use YOLO-World style text prompts for zero-shot detection",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Text prompt for the target person (repeat to add multiple)",
    )
    parser.add_argument(
        "--alert-message",
        default="⚠️ 駐車監視員を検知しました！",
        help="Alert message to print when a target is inside the zone",
    )
    parser.add_argument(
        "--dwell-seconds",
        type=float,
        default=5.0,
        help="Seconds a target must stay in the zone before the alarm",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zone_coords = args.zone or (50, 50, 300, 300)
    zone = AlertZone(*zone_coords)

    default_prompts = [
        "Parking enforcement officer in Japan wearing green or lime uniform and fluorescent reflective safety vest",
        "Officer with peaked cap or work cap checking license plate area of parked car",
        "Officer carrying shoulder bag and holding handheld terminal or smartphone near car front",
    ]
    text_prompts: Optional[List[str]] = None
    if args.use_prompts or args.prompts:
        text_prompts = args.prompts or default_prompts

    alert_system = ParkingAlert(
        model_name=args.model,
        zone=zone,
        score_threshold=args.threshold,
        alert_cooldown=args.cooldown,
        text_prompts=text_prompts,
        imgsz=args.imgsz,
        alert_message=args.alert_message,
        dwell_seconds=args.dwell_seconds,
    )

    print("--- 検知設定 ---")
    print(f"モデル: {args.model}")
    
    # モデルがトレーニング済みかチェック
    model_path = Path(args.model)
    is_trained_model = model_path.exists() and "parking_officer" in str(model_path)
    
    if is_trained_model:
        print("検出モード: トレーニング済み駐車監視員検出モデル")
    elif text_prompts:
        print("検出モード: YOLO-World (テキストプロンプト)")
    else:
        print("検出モード: COCOのpersonクラス")
    if text_prompts:
        print("使用プロンプト:")
        for prompt in text_prompts:
            print(f"  - {prompt}")
    print(f"アラートゾーン: {zone_coords}")
    print(f"信頼度しきい値: {args.threshold}")
    print(f"アラートメッセージ: {args.alert_message}")
    print(f"アラーム条件: 監視エリアに {args.dwell_seconds} 秒以上連続滞在")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Unable to open video source: {args.source}")
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received, exiting...")
                break

            frame = cv2.resize(frame, (args.imgsz, args.imgsz))
            annotated = alert_system.process_frame(frame)
            cv2.imshow("YOLO Parking Alert", annotated)

            key = cv2.waitKey(1)
            if key == 27:  # ESC to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
