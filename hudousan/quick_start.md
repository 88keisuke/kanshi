# クイックスタートガイド

## 最も簡単な方法: 継続的トレーニングを開始

```bash
# 1. スクリプトに実行権限を付与（初回のみ）
chmod +x start_training.sh

# 2. 継続的トレーニングを開始
./start_training.sh
```

このスクリプトは以下を自動的に実行します:
- ネットから駐車監視員の画像を収集
- データセットを準備
- モデルをトレーニング
- 評価と改善を繰り返し

## 手動でステップごとに実行する場合

### ステップ1: 画像収集
```bash
python3 collect_images.py --max-images 100
```

### ステップ2: 画像アノテーション（重要！）
```bash
python3 annotate_images.py
```
マウスで駐車監視員を囲んでラベルを付けます。

### ステップ3: モデルトレーニング
```bash
python3 train_model.py --split-dataset --epochs 100
```

### ステップ4: 検出の実行
```bash
python3 clone_home.py --model runs/detect/parking_officer/weights/best.pt --source 0
```

## 必要なパッケージのインストール

### 自動インストール（推奨）

```bash
./install_dependencies.sh
```

### 手動インストール

```bash
pip install -r requirements.txt
```

### 依存関係の確認

```bash
python3 check_dependencies.py
```

## 注意事項

- 画像アノテーションは正確に行ってください（精度に直接影響します）
- 最低50枚以上の画像を収集してください
- トレーニングには時間がかかります（GPU使用を推奨）

