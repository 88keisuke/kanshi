# -*- coding: utf-8 -*-
"""
画像アノテーションツール

収集した画像にバウンディングボックスを描画して、
YOLO形式のラベルファイルを生成します。
"""

import argparse
import cv2
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import json


class ImageAnnotator:
    """画像アノテーションクラス"""
    
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像ファイルのリスト
        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) + 
            list(self.images_dir.glob("*.png"))
        )
        self.current_index = 0
        
        # アノテーション状態
        self.drawing = False
        self.start_point: Optional[Tuple[int, int]] = None
        self.end_point: Optional[Tuple[int, int]] = None
        self.current_boxes: List[Tuple[int, int, int, int]] = []
        
        # クラスID（駐車監視員 = 0）
        self.class_id = 0
    
    def load_image(self, index: int) -> Optional[Tuple[cv2.Mat, str]]:
        """画像を読み込み"""
        if 0 <= index < len(self.image_files):
            img_path = self.image_files[index]
            img = cv2.imread(str(img_path))
            if img is not None:
                return img, img_path.stem
        return None, None
    
    def mouse_callback(self, event, x, y, flags, param):
        """マウスイベントコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                # 正規化
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                self.current_boxes.append((x1, y1, x2, y2))
                self.start_point = None
                self.end_point = None
    
    def draw_boxes(self, img, boxes: List[Tuple[int, int, int, int]]) -> cv2.Mat:
        """バウンディングボックスを描画"""
        img_copy = img.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 現在描画中のボックス
        if self.start_point and self.end_point:
            cv2.rectangle(img_copy, self.start_point, self.end_point, (0, 0, 255), 2)
        
        return img_copy
    
    def save_label(self, image_name: str, boxes: List[Tuple[int, int, int, int]], img_shape: Tuple[int, int]):
        """YOLO形式のラベルファイルを保存"""
        label_path = self.labels_dir / f"{image_name}.txt"
        h, w = img_shape[:2]
        
        with open(label_path, 'w') as f:
            for x1, y1, x2, y2 in boxes:
                # YOLO形式に変換（正規化座標）
                center_x = ((x1 + x2) / 2.0) / w
                center_y = ((y1 + y2) / 2.0) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                f.write(f"{self.class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"ラベルを保存しました: {label_path}")
    
    def load_label(self, image_name: str, img_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """YOLO形式のラベルファイルを読み込み"""
        label_path = self.labels_dir / f"{image_name}.txt"
        boxes = []
        
        if label_path.exists():
            h, w = img_shape[:2]
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1]) * w
                        center_y = float(parts[2]) * h
                        width = float(parts[3]) * w
                        height = float(parts[4]) * h
                        
                        x1 = int(center_x - width / 2)
                        y1 = int(center_y - height / 2)
                        x2 = int(center_x + width / 2)
                        y2 = int(center_y + height / 2)
                        
                        boxes.append((x1, y1, x2, y2))
        
        return boxes
    
    def annotate(self):
        """アノテーションループ"""
        if len(self.image_files) == 0:
            print(f"エラー: {self.images_dir} に画像が見つかりません")
            return
        
        print("画像アノテーションツール")
        print("操作方法:")
        print("  マウスドラッグ: バウンディングボックスを描画")
        print("  's': 保存して次の画像へ")
        print("  'd': 最後のボックスを削除")
        print("  'c': すべてのボックスをクリア")
        print("  'n': 次の画像へ（保存しない）")
        print("  'p': 前の画像へ")
        print("  'q': 終了")
        print(f"\n総画像数: {len(self.image_files)}")
        print()
        
        while self.current_index < len(self.image_files):
            img, image_name = self.load_image(self.current_index)
            if img is None:
                break
            
            # 既存のラベルを読み込み
            self.current_boxes = self.load_label(image_name, img.shape)
            
            window_name = f"Annotation Tool - {self.current_index + 1}/{len(self.image_files)}"
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, self.mouse_callback)
            
            while True:
                display_img = self.draw_boxes(img, self.current_boxes)
                
                # 情報を表示
                info_text = f"Image {self.current_index + 1}/{len(self.image_files)} | Boxes: {len(self.current_boxes)}"
                cv2.putText(
                    display_img,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                
                cv2.imshow(window_name, display_img)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # 保存
                    if self.current_boxes:
                        self.save_label(image_name, self.current_boxes, img.shape)
                    else:
                        print("警告: ボックスがありません。スキップします。")
                    self.current_index += 1
                    self.current_boxes = []
                    self.start_point = None
                    self.end_point = None
                    break
                
                elif key == ord('d'):  # 最後のボックスを削除
                    if self.current_boxes:
                        self.current_boxes.pop()
                        print(f"ボックスを削除しました。残り: {len(self.current_boxes)}")
                
                elif key == ord('c'):  # すべてクリア
                    self.current_boxes = []
                    self.start_point = None
                    self.end_point = None
                    print("すべてのボックスをクリアしました")
                
                elif key == ord('n'):  # 次の画像（保存しない）
                    self.current_index += 1
                    self.current_boxes = []
                    self.start_point = None
                    self.end_point = None
                    break
                
                elif key == ord('p'):  # 前の画像
                    if self.current_index > 0:
                        self.current_index -= 1
                        self.current_boxes = []
                        self.start_point = None
                        self.end_point = None
                        break
                
                elif key == ord('q'):  # 終了
                    cv2.destroyAllWindows()
                    print("\nアノテーションを終了しました")
                    return
            
            cv2.destroyWindow(window_name)
        
        print("\nすべての画像のアノテーションが完了しました！")


def main():
    parser = argparse.ArgumentParser(description="画像アノテーションツール")
    parser.add_argument(
        "--images-dir",
        default="dataset/images",
        help="画像ディレクトリ (default: dataset/images)",
    )
    parser.add_argument(
        "--labels-dir",
        default="dataset/labels",
        help="ラベルディレクトリ (default: dataset/labels)",
    )
    
    args = parser.parse_args()
    
    annotator = ImageAnnotator(args.images_dir, args.labels_dir)
    annotator.annotate()


if __name__ == "__main__":
    main()

