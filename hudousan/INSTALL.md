# インストールガイド

## クイックインストール

```bash
# 1. 依存パッケージをインストール
./install_dependencies.sh

# 2. 依存関係を確認
python3 check_dependencies.py
```

## 詳細なインストール手順

### 1. Pythonの確認

Python 3.7以上が必要です。

```bash
python3 --version
```

### 2. 依存パッケージのインストール

#### 方法A: 自動インストールスクリプト（推奨）

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

#### 方法B: requirements.txtを使用

```bash
pip install -r requirements.txt
```

#### 方法C: 個別インストール

```bash
pip install ultralytics opencv-python requests beautifulsoup4 selenium pyyaml
```

### 3. 依存関係の確認

```bash
python3 check_dependencies.py
```

不足しているパッケージがある場合は、自動インストール:

```bash
python3 check_dependencies.py --auto-install
```

### 4. ChromeDriverのインストール（画像収集機能を使用する場合）

#### macOS

```bash
brew install chromedriver
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get install chromium-chromedriver
```

#### Windows

1. [ChromeDriver公式サイト](https://chromedriver.chromium.org/)からダウンロード
2. PATHに追加

### 5. 動作確認

各スクリプトを実行して、依存パッケージが正しくインストールされているか確認:

```bash
# 検出スクリプト
python3 clone_home.py --help

# 画像収集スクリプト
python3 collect_images.py --help

# トレーニングスクリプト
python3 train_model.py --help
```

## トラブルシューティング

### パッケージのインストールエラー

```bash
# pipをアップグレード
pip install --upgrade pip

# 再度インストール
pip install -r requirements.txt
```

### 権限エラー

```bash
# ユーザー領域にインストール
pip install --user -r requirements.txt
```

### 仮想環境の使用（推奨）

```bash
# 仮想環境を作成
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate  # Windows

# 依存パッケージをインストール
pip install -r requirements.txt
```

## 必要なパッケージ一覧

- **ultralytics**: YOLOモデルの実行
- **opencv-python**: 画像処理とビデオキャプチャ
- **requests**: HTTPリクエスト（画像収集）
- **beautifulsoup4**: HTMLパース（画像収集）
- **selenium**: ブラウザ自動化（画像収集、オプション）
- **pyyaml**: YAMLファイルの読み書き（データセット設定）

## システム要件

- Python 3.7以上
- メモリ: 4GB以上（推奨8GB以上）
- ストレージ: 2GB以上の空き容量
- GPU: オプション（トレーニングを高速化）

