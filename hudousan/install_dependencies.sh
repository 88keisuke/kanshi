#!/bin/bash
# 依存パッケージをインストールするスクリプト

set -e  # エラーが発生したら停止

echo "=========================================="
echo "依存パッケージのインストール"
echo "=========================================="
echo ""

# Pythonのバージョンチェック
echo "Pythonのバージョンを確認しています..."
if ! command -v python3 &> /dev/null; then
    echo "エラー: python3が見つかりません"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION が見つかりました"
echo ""

# pipの確認
echo "pipのバージョンを確認しています..."
if ! command -v pip3 &> /dev/null; then
    echo "警告: pip3が見つかりません。get-pip.pyを使用します..."
    python3 -m ensurepip --upgrade || {
        echo "エラー: pipをインストールできませんでした"
        exit 1
    }
fi

PIP_VERSION=$(pip3 --version 2>&1 | awk '{print $2}')
echo "✓ pip $PIP_VERSION が見つかりました"
echo ""

# requirements.txtの存在確認
if [ ! -f "requirements.txt" ]; then
    echo "エラー: requirements.txtが見つかりません"
    exit 1
fi

# パッケージのインストール
echo "パッケージをインストールしています..."
echo ""

pip3 install --upgrade pip
pip3 install -r requirements.txt

echo ""
echo "=========================================="
echo "インストール完了！"
echo "=========================================="
echo ""

# インストール確認
echo "インストールされたパッケージを確認しています..."
echo ""

python3 -c "import ultralytics; print('✓ ultralytics')" 2>/dev/null || echo "✗ ultralytics (インストール失敗)"
python3 -c "import cv2; print('✓ opencv-python')" 2>/dev/null || echo "✗ opencv-python (インストール失敗)"
python3 -c "import requests; print('✓ requests')" 2>/dev/null || echo "✗ requests (インストール失敗)"
python3 -c "import bs4; print('✓ beautifulsoup4')" 2>/dev/null || echo "✗ beautifulsoup4 (インストール失敗)"
python3 -c "import selenium; print('✓ selenium')" 2>/dev/null || echo "✗ selenium (インストール失敗)"
python3 -c "import yaml; print('✓ pyyaml')" 2>/dev/null || echo "✗ pyyaml (インストール失敗)"

echo ""
echo "注意: ChromeDriverが必要な場合:"
echo "  macOS: brew install chromedriver"
echo "  Linux: sudo apt-get install chromium-chromedriver"
echo "  Windows: https://chromedriver.chromium.org/ からダウンロード"

