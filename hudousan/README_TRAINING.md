# 駐車監視員検出システム - 機械学習トレーニングガイド

このシステムは、指定された特徴を持つ駐車監視員のみを検出するためのカスタムYOLOモデルをトレーニングします。

## 特徴

- **指定された特徴のみを検出**: 以下の特徴をすべて満たす人物のみを検出
  - 緑色または黄緑色の制服
  - 蛍光色の反射安全ベスト
  - 制帽または作業用キャップ
  - 肩掛けの業務用バッグ
  - 手にハンディ端末、スマートフォン、または撮影機器
  - 路上に駐車されている車両のフロント付近またはナンバープレート周辺を確認する動作

- **継続的トレーニング**: ネットから画像を収集し、自動的にモデルを改善
- **カスタムモデル**: 汎用のperson検出ではなく、駐車監視員専用のモデル

## セットアップ

### 依存パッケージのインストール

#### 自動インストール（推奨）

```bash
# インストールスクリプトを実行
./install_dependencies.sh
```

または、Pythonスクリプトを使用:

```bash
# 依存関係をチェックして自動インストール
python3 check_dependencies.py --auto-install
```

#### 手動インストール

```bash
pip install -r requirements.txt
```

または個別にインストール:

```bash
pip install ultralytics opencv-python requests beautifulsoup4 selenium pyyaml
```

### ChromeDriverのインストール（画像収集に必要）

画像収集機能を使用する場合は、ChromeDriverが必要です。

- macOS: `brew install chromedriver`
- Linux: `sudo apt-get install chromium-chromedriver` またはパッケージマネージャーからインストール
- Windows: [ChromeDriver公式サイト](https://chromedriver.chromium.org/)からダウンロード

### 依存関係の確認

```bash
# 依存パッケージの状態を確認
python3 check_dependencies.py
```

## 使用方法

### 1. 画像収集

ネットから駐車監視員の画像を収集します。

```bash
python collect_images.py --output-dir dataset/images --max-images 100
```

### 2. 画像アノテーション

収集した画像にバウンディングボックスを描画してラベルを作成します。

```bash
python annotate_images.py --images-dir dataset/images --labels-dir dataset/labels
```

**操作方法:**
- マウスドラッグ: バウンディングボックスを描画
- `s`: 保存して次の画像へ
- `d`: 現在のボックスを削除
- `c`: すべてのボックスをクリア
- `n`: 次の画像へ（保存しない）
- `p`: 前の画像へ
- `q`: 終了

### 3. モデルトレーニング

アノテーション済みの画像でモデルをトレーニングします。

```bash
python train_model.py \
    --model yolov8n.pt \
    --dataset-dir dataset \
    --epochs 100 \
    --batch 16 \
    --split-dataset
```

### 4. 継続的トレーニング（推奨）

画像収集、トレーニング、評価を自動的に繰り返してモデルを改善します。

```bash
python continuous_training.py \
    --dataset-dir dataset \
    --model-dir models \
    --min-images 50 \
    --target-accuracy 0.85 \
    --max-iterations 10
```

このスクリプトは以下を自動的に実行します:
1. ネットから画像を収集
2. データセットを準備
3. モデルをトレーニング
4. モデルを評価
5. 目標精度に達するまで繰り返し

### 5. トレーニング済みモデルを使用

トレーニング済みのモデルを使用して検出を実行します。

```bash
python clone_home.py \
    --model models/parking_officer_iter001.pt \
    --source 0 \
    --zone 50 50 300 300 \
    --threshold 0.5
```

## ディレクトリ構造

```
hudousan/
├── clone_home.py              # メイン検出スクリプト
├── collect_images.py          # 画像収集スクリプト
├── annotate_images.py          # アノテーションツール
├── train_model.py             # モデルトレーニングスクリプト
├── continuous_training.py     # 継続的トレーニングスクリプト
├── dataset/                    # データセット
│   ├── images/                # 収集した画像
│   ├── labels/                # アノテーション（YOLO形式）
│   ├── train/                 # 訓練データ（自動生成）
│   │   ├── images/
│   │   └── labels/
│   └── val/                   # 検証データ（自動生成）
│       ├── images/
│       └── labels/
├── models/                     # トレーニング済みモデル
│   ├── parking_officer_iter001.pt
│   ├── parking_officer_iter002.pt
│   └── training_history.json
└── runs/                       # トレーニング結果
    └── detect/
        └── iteration_XXX/
```

## トレーニングの流れ

1. **画像収集**: `collect_images.py`でネットから画像を収集
2. **アノテーション**: `annotate_images.py`で画像にラベルを付与
3. **データセット準備**: `train_model.py --split-dataset`で訓練/検証に分割
4. **トレーニング**: `train_model.py`でモデルをトレーニング
5. **評価**: 検証データで精度を確認
6. **改善**: 精度が不十分な場合は、より多くの画像を収集して再トレーニング

または、`continuous_training.py`を使用して上記の流れを自動化できます。

## 注意事項

- 画像収集は、利用規約を遵守してください
- アノテーションは正確に行ってください（精度に直接影響します）
- 十分な数の画像（最低50枚以上、推奨100枚以上）を収集してください
- トレーニングには時間がかかります（GPU使用を推奨）

## トラブルシューティング

### 画像収集が動作しない
- ChromeDriverが正しくインストールされているか確認
- Seleniumがインストールされているか確認: `pip install selenium`

### トレーニングエラー
- データセットディレクトリが正しく設定されているか確認
- 画像とラベルのファイル名が一致しているか確認
- ラベルファイルがYOLO形式（正規化座標）になっているか確認

### 検出精度が低い
- より多くの画像を収集
- アノテーションの精度を確認
- トレーニングエポック数を増やす
- データ拡張を有効にする

