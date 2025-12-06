#!/bin/bash
# 駐車監視員検出モデルの継続的トレーニングを開始するスクリプト

echo "=========================================="
echo "駐車監視員検出モデル - 継続的トレーニング"
echo "=========================================="
echo ""

# 必要なパッケージのチェック
echo "依存パッケージをチェックしています..."
python3 -c "import ultralytics" 2>/dev/null || {
    echo "エラー: ultralyticsがインストールされていません"
    echo "実行: pip install ultralytics"
    exit 1
}

python3 -c "import cv2" 2>/dev/null || {
    echo "エラー: opencv-pythonがインストールされていません"
    echo "実行: pip install opencv-python"
    exit 1
}

python3 -c "import requests" 2>/dev/null || {
    echo "エラー: requestsがインストールされていません"
    echo "実行: pip install requests"
    exit 1
}

echo "✓ すべての依存パッケージがインストールされています"
echo ""

# ディレクトリの作成
mkdir -p dataset/images
mkdir -p dataset/labels
mkdir -p models

echo "継続的トレーニングを開始します..."
echo "  データセット: dataset/"
echo "  モデル保存先: models/"
echo ""
echo "注意: このプロセスは時間がかかります"
echo "      Ctrl+Cで中断できます"
echo ""

# 継続的トレーニングを実行
python3 continuous_training.py \
    --dataset-dir dataset \
    --model-dir models \
    --min-images 50 \
    --target-accuracy 0.85 \
    --max-iterations 10

echo ""
echo "トレーニングが完了しました！"
echo "ベストモデル: models/parking_officer_iterXXX.pt"
echo ""
echo "使用方法:"
echo "  python3 clone_home.py --model models/parking_officer_iterXXX.pt --source 0"

