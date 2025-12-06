# -*- coding: utf-8 -*-
"""
駐車監視員検出用YOLOモデルのトレーニングスクリプト

収集した画像とアノテーションを使用してYOLOモデルをファインチューニングします。
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# 依存パッケージのチェック
try:
    import yaml
except ImportError:
    print("エラー: pyyamlがインストールされていません")
    print("インストール: pip install pyyaml")
    sys.exit(1)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("エラー: ultralyticsがインストールされていません")
    print("インストール: pip install ultralytics")
    print()
    print("または、すべての依存パッケージをインストール:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def create_dataset_yaml(dataset_dir: str, output_yaml: str = "dataset.yaml") -> str:
    """YOLO用のデータセット設定ファイルを作成"""
    dataset_path = Path(dataset_dir).absolute()
    
    # ディレクトリ構造を確認
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    val_images = dataset_path / "val" / "images"
    val_labels = dataset_path / "val" / "labels"
    
    # ディレクトリが存在しない場合は作成
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)
    
    # データセット設定
    dataset_config = {
        'path': str(dataset_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': None,
        'nc': 1,  # クラス数（駐車監視員のみ）
        'names': {
            0: 'parking_officer'
        }
    }
    
    yaml_path = Path(output_yaml)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"データセット設定ファイルを作成しました: {yaml_path}")
    return str(yaml_path)


def split_dataset(images_dir: str, labels_dir: str, train_ratio: float = 0.8):
    """データセットを訓練用と検証用に分割"""
    from pathlib import Path
    import shutil
    import random
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # 画像ファイルを取得
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    random.shuffle(image_files)
    
    # 分割
    split_idx = int(len(image_files) * train_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    # 出力ディレクトリ
    dataset_root = images_path.parent.parent
    train_img_dir = dataset_root / "train" / "images"
    train_lbl_dir = dataset_root / "train" / "labels"
    val_img_dir = dataset_root / "val" / "images"
    val_lbl_dir = dataset_root / "val" / "labels"
    
    for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 訓練データをコピー
    for img_file in train_images:
        lbl_file = labels_path / (img_file.stem + ".txt")
        if lbl_file.exists():
            shutil.copy2(img_file, train_img_dir / img_file.name)
            shutil.copy2(lbl_file, train_lbl_dir / lbl_file.name)
    
    # 検証データをコピー
    for img_file in val_images:
        lbl_file = labels_path / (img_file.stem + ".txt")
        if lbl_file.exists():
            shutil.copy2(img_file, val_img_dir / img_file.name)
            shutil.copy2(lbl_file, val_lbl_dir / lbl_file.name)
    
    print(f"データセット分割完了:")
    print(f"  訓練データ: {len(train_images)} 枚")
    print(f"  検証データ: {len(val_images)} 枚")


def train(
    model_name: str = "yolov8n.pt",
    dataset_yaml: str = "dataset.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    project: str = "runs/detect",
    name: str = "parking_officer",
) -> str:
    """YOLOモデルをトレーニング"""
    
    if not ULTRALYTICS_AVAILABLE:
        print("エラー: ultralyticsがインストールされていません。")
        sys.exit(1)
    
    print(f"モデルトレーニングを開始します...")
    print(f"  ベースモデル: {model_name}")
    print(f"  データセット: {dataset_yaml}")
    print(f"  エポック数: {epochs}")
    print(f"  画像サイズ: {imgsz}")
    print(f"  バッチサイズ: {batch}")
    
    # モデルをロード
    model = YOLO(model_name)
    
    # トレーニング実行
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=50,  # Early stopping patience
        save=True,
        plots=True,
        verbose=True,
    )
    
    # ベストモデルのパスを返す
    best_model_path = Path(project) / name / "weights" / "best.pt"
    if best_model_path.exists():
        print(f"\nトレーニング完了！")
        print(f"ベストモデル: {best_model_path}")
        return str(best_model_path)
    else:
        print(f"\n警告: ベストモデルが見つかりませんでした。")
        return ""


def main():
    parser = argparse.ArgumentParser(description="駐車監視員検出用YOLOモデルのトレーニング")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="ベースモデル (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="データセットディレクトリ (default: dataset)",
    )
    parser.add_argument(
        "--dataset-yaml",
        help="データセットYAMLファイル（指定しない場合は自動生成）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="トレーニングエポック数 (default: 100)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="画像サイズ (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="バッチサイズ (default: 16)",
    )
    parser.add_argument(
        "--device",
        help="デバイス (cpu, 0, 1, ...) 指定しない場合は自動",
    )
    parser.add_argument(
        "--project",
        default="runs/detect",
        help="プロジェクトディレクトリ (default: runs/detect)",
    )
    parser.add_argument(
        "--name",
        default="parking_officer",
        help="実行名 (default: parking_officer)",
    )
    parser.add_argument(
        "--split-dataset",
        action="store_true",
        help="データセットを訓練/検証に分割",
    )
    
    args = parser.parse_args()
    
    # データセットYAMLファイルの準備
    if args.dataset_yaml:
        dataset_yaml = args.dataset_yaml
    else:
        # データセット分割が必要な場合
        if args.split_dataset:
            images_dir = Path(args.dataset_dir) / "images"
            labels_dir = Path(args.dataset_dir) / "labels"
            if images_dir.exists() and labels_dir.exists():
                split_dataset(str(images_dir), str(labels_dir))
        
        dataset_yaml = create_dataset_yaml(args.dataset_dir)
    
    # トレーニング実行
    best_model = train(
        model_name=args.model,
        dataset_yaml=dataset_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    
    if best_model:
        print(f"\n使用例:")
        print(f"  python clone_home.py --model {best_model}")


if __name__ == "__main__":
    main()

