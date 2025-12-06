# -*- coding: utf-8 -*-
"""
継続的トレーニングシステム

画像を収集し、モデルをトレーニングし、評価を繰り返して
駐車監視員検出モデルを継続的に改善します。
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# 依存パッケージのチェック
try:
    from collect_images import ImageCollector
    from train_model import train, create_dataset_yaml, split_dataset
except ImportError as e:
    print(f"エラー: 必要なモジュールをインポートできません: {e}")
    print()
    print("依存パッケージをインストールしてください:")
    print("  pip install -r requirements.txt")
    print()
    print("または、自動インストールスクリプトを実行:")
    print("  ./install_dependencies.sh")
    sys.exit(1)


class ContinuousTrainer:
    """継続的トレーニングクラス"""
    
    def __init__(
        self,
        dataset_dir: str = "dataset",
        model_dir: str = "models",
        min_images: int = 50,
        target_accuracy: float = 0.85,
        max_iterations: int = 10,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.min_images = min_images
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.model_dir / "training_history.json"
        self.history = self.load_history()
    
    def load_history(self) -> Dict:
        """トレーニング履歴を読み込み"""
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'iterations': [],
            'best_model': None,
            'best_accuracy': 0.0,
        }
    
    def save_history(self):
        """トレーニング履歴を保存"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def collect_images(self, num_images: int = 50) -> int:
        """画像を収集"""
        print(f"\n{'='*60}")
        print(f"画像収集フェーズ")
        print(f"{'='*60}")
        
        images_dir = self.dataset_dir / "images"
        collector = ImageCollector(
            output_dir=str(images_dir),
            max_images=num_images,
        )
        
        collected = collector.collect_all()
        return collected
    
    def prepare_dataset(self):
        """データセットを準備（分割）"""
        print(f"\n{'='*60}")
        print(f"データセット準備フェーズ")
        print(f"{'='*60}")
        
        images_dir = self.dataset_dir / "images"
        labels_dir = self.dataset_dir / "labels"
        
        if not images_dir.exists():
            print(f"エラー: 画像ディレクトリが見つかりません: {images_dir}")
            return False
        
        # ラベルディレクトリが存在しない場合は作成
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # データセット分割
        split_dataset(str(images_dir), str(labels_dir))
        
        # データセットYAMLを作成
        dataset_yaml = create_dataset_yaml(str(self.dataset_dir))
        return dataset_yaml
    
    def train_model(self, iteration: int, base_model: Optional[str] = None) -> Dict:
        """モデルをトレーニング"""
        print(f"\n{'='*60}")
        print(f"モデルトレーニングフェーズ (イテレーション {iteration})")
        print(f"{'='*60}")
        
        # ベースモデルの決定
        if base_model is None:
            if self.history['best_model'] and Path(self.history['best_model']).exists():
                base_model = self.history['best_model']
                print(f"前回のベストモデルを使用: {base_model}")
            else:
                base_model = "yolov8n.pt"
                print(f"デフォルトモデルを使用: {base_model}")
        
        # データセット準備
        dataset_yaml = self.prepare_dataset()
        if not dataset_yaml:
            return {'success': False, 'error': 'データセット準備に失敗'}
        
        # トレーニング実行
        project_name = f"iteration_{iteration:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            best_model_path = train(
                model_name=base_model,
                dataset_yaml=dataset_yaml,
                epochs=100,
                imgsz=640,
                batch=16,
                project=str(self.model_dir / "runs"),
                name=project_name,
            )
            
            if not best_model_path:
                return {'success': False, 'error': 'トレーニングに失敗'}
            
            # モデルを保存
            saved_model_path = self.model_dir / f"parking_officer_iter{iteration:03d}.pt"
            import shutil
            shutil.copy2(best_model_path, saved_model_path)
            
            # メトリクスを読み込み（簡易版）
            metrics = self.extract_metrics(best_model_path)
            
            return {
                'success': True,
                'model_path': str(saved_model_path),
                'best_model_path': best_model_path,
                'metrics': metrics,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def extract_metrics(self, model_path: str) -> Dict:
        """トレーニング結果からメトリクスを抽出"""
        import csv
        
        # results.csvを探す
        model_dir = Path(model_path).parent
        results_csv = model_dir / "results.csv"
        
        metrics = {
            'mAP50': 0.0,
            'mAP50-95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
        }
        
        if results_csv.exists():
            try:
                with open(results_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        # 最後のエポックの結果を使用
                        last_row = rows[-1]
                        metrics['mAP50'] = float(last_row.get('metrics/mAP50(B)', 0.0))
                        metrics['mAP50-95'] = float(last_row.get('metrics/mAP50-95(B)', 0.0))
                        metrics['precision'] = float(last_row.get('metrics/precision(B)', 0.0))
                        metrics['recall'] = float(last_row.get('metrics/recall(B)', 0.0))
            except Exception as e:
                print(f"メトリクス読み込みエラー: {e}")
        
        return metrics
    
    def evaluate_model(self, model_path: str) -> float:
        """モデルを評価"""
        try:
            from ultralytics import YOLO
            
            if not Path(model_path).exists():
                return 0.0
            
            # データセットYAMLを取得
            dataset_yaml = self.dataset_dir / "dataset.yaml"
            if not dataset_yaml.exists():
                dataset_yaml = create_dataset_yaml(str(self.dataset_dir))
            
            # モデルをロード
            model = YOLO(model_path)
            
            # 検証を実行
            results = model.val(data=str(dataset_yaml), verbose=False)
            
            # mAP50を精度として使用
            if hasattr(results, 'box') and hasattr(results.box, 'map50'):
                accuracy = float(results.box.map50)
            elif hasattr(results, 'results_dict'):
                accuracy = float(results.results_dict.get('metrics/mAP50(B)', 0.0))
            else:
                # フォールバック: メトリクスから取得
                metrics = self.extract_metrics(model_path)
                accuracy = metrics.get('mAP50', 0.0)
            
            return accuracy
        except Exception as e:
            print(f"評価エラー: {e}")
            # フォールバック: メトリクスから取得
            metrics = self.extract_metrics(model_path)
            return metrics.get('mAP50', 0.0)
    
    def run_iteration(self, iteration: int) -> bool:
        """1回のイテレーションを実行"""
        print(f"\n{'#'*60}")
        print(f"イテレーション {iteration}/{self.max_iterations}")
        print(f"{'#'*60}")
        
        # 1. 画像収集
        collected = self.collect_images(num_images=50)
        if collected < self.min_images:
            print(f"警告: 収集画像数が少なすぎます ({collected} < {self.min_images})")
            print("続行しますが、精度に影響する可能性があります。")
        
        # 2. モデルトレーニング
        result = self.train_model(iteration)
        if not result['success']:
            print(f"エラー: トレーニングに失敗しました: {result.get('error', 'Unknown error')}")
            return False
        
        # 3. 評価
        accuracy = self.evaluate_model(result['model_path'])
        print(f"評価結果: 精度 = {accuracy:.3f}")
        
        # 4. 履歴を更新
        iteration_data = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'collected_images': collected,
            'model_path': result['model_path'],
            'accuracy': accuracy,
            'metrics': result.get('metrics', {}),
        }
        self.history['iterations'].append(iteration_data)
        
        # ベストモデルを更新
        if accuracy > self.history['best_accuracy']:
            self.history['best_accuracy'] = accuracy
            self.history['best_model'] = result['model_path']
            print(f"🎉 新しいベストモデル！精度: {accuracy:.3f}")
        
        self.save_history()
        
        # 目標精度に達したかチェック
        if accuracy >= self.target_accuracy:
            print(f"\n✅ 目標精度 ({self.target_accuracy}) に到達しました！")
            return True
        
        return False
    
    def run(self):
        """継続的トレーニングを実行"""
        print(f"継続的トレーニングを開始します")
        print(f"  データセットディレクトリ: {self.dataset_dir}")
        print(f"  モデルディレクトリ: {self.model_dir}")
        print(f"  最小画像数: {self.min_images}")
        print(f"  目標精度: {self.target_accuracy}")
        print(f"  最大イテレーション数: {self.max_iterations}")
        
        start_iteration = len(self.history['iterations']) + 1
        
        for iteration in range(start_iteration, start_iteration + self.max_iterations):
            success = self.run_iteration(iteration)
            
            if success:
                print(f"\n✅ 目標達成！トレーニングを終了します。")
                break
            
            if iteration < start_iteration + self.max_iterations - 1:
                print(f"\n次のイテレーションまで待機します...")
                time.sleep(5)  # 実際の運用では、より長い待機時間を設定
        
        print(f"\n{'='*60}")
        print(f"トレーニング完了")
        print(f"{'='*60}")
        print(f"ベストモデル: {self.history['best_model']}")
        print(f"ベスト精度: {self.history['best_accuracy']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="継続的トレーニングシステム")
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="データセットディレクトリ (default: dataset)",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="モデルディレクトリ (default: models)",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=50,
        help="最小画像数 (default: 50)",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.85,
        help="目標精度 (default: 0.85)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="最大イテレーション数 (default: 10)",
    )
    
    args = parser.parse_args()
    
    trainer = ContinuousTrainer(
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        min_images=args.min_images,
        target_accuracy=args.target_accuracy,
        max_iterations=args.max_iterations,
    )
    
    trainer.run()


if __name__ == "__main__":
    main()

