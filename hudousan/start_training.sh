#!/bin/bash
# 駐車監視員検出モデルの継続的トレーニングを開始するスクリプト

echo "=========================================="
echo "駐車監視員検出モデル - 継続的トレーニング"
echo "=========================================="
echo ""

# 依存パッケージのチェックとインストール
echo "依存パッケージをチェックしています..."
if [ ! -f "requirements.txt" ]; then
    echo "エラー: requirements.txtが見つかりません"
    exit 1
fi

# check_dependencies.pyを使用してチェック
if python3 check_dependencies.py 2>/dev/null; then
    echo "✓ すべての依存パッケージがインストールされています"
    echo ""
else
    echo ""
    echo "不足しているパッケージを自動インストールしますか？ (y/n)"
    read -r response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo ""
        python3 check_dependencies.py --auto-install || {
            echo ""
            echo "自動インストールに失敗しました。手動でインストールしてください:"
            echo "  ./install_dependencies.sh"
            exit 1
        }
        echo ""
    else
        echo ""
        echo "依存パッケージをインストールしてください:"
        echo "  ./install_dependencies.sh"
        echo "  または"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
fi

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

