# -*- coding: utf-8 -*-
"""
駐車監視員の画像をネットから収集するスクリプト

Google画像検索やその他のソースから駐車監視員の画像を収集し、
トレーニング用データセットを構築します。
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote
import json

# 依存パッケージのチェック
try:
    import requests
except ImportError:
    print("エラー: requestsがインストールされていません")
    print("インストール: pip install requests")
    sys.exit(1)

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("警告: seleniumがインストールされていません。pip install selenium を実行してください。")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("警告: beautifulsoup4がインストールされていません。pip install beautifulsoup4 を実行してください。")


class ImageCollector:
    """画像収集クラス"""
    
    def __init__(self, output_dir: str = "dataset/images", max_images: int = 100, min_images: int = 50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_images = max_images
        self.min_images = min_images
        self.collected_count = 0
        
        # 検索クエリ（日本語と英語）
        self.search_queries = [
            "駐車監視員 日本",
            "駐車違反取締員",
            "駐車監視員 制服",
            "parking enforcement officer Japan",
            "駐車監視員 緑色 制服",
            "駐車監視員 反射ベスト",
            "parking officer green uniform",
            "駐車監視員 制帽",
            "駐車監視員 ハンディ端末",
        ]
    
    def download_image(self, url: str, filename: str) -> bool:
        """画像をダウンロード"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10, stream=True)
            response.raise_for_status()
            
            # 画像ファイルかチェック
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False
            
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # ファイルサイズチェック（小さすぎるファイルはスキップ）
            if filepath.stat().st_size < 1000:  # 1KB未満
                filepath.unlink()
                return False
            
            return True
        except Exception as e:
            print(f"画像ダウンロードエラー ({url}): {e}")
            return False
    
    def collect_from_duckduckgo(self, query: str, num_images: int = 20) -> int:
        """DuckDuckGo画像検索から画像を収集（API不要）"""
        if not SELENIUM_AVAILABLE:
            print("警告: Seleniumが利用できません。pip install selenium を実行してください。")
            print("      Seleniumなしでも他の方法で画像収集を試みます。")
            return 0
        
        collected = 0
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            
            driver = webdriver.Chrome(options=chrome_options)
            
            # DuckDuckGo画像検索
            search_url = f"https://duckduckgo.com/?q={quote(query)}&iax=images&ia=images"
            driver.get(search_url)
            time.sleep(2)
            
            # 画像要素を取得
            images = driver.find_elements(By.CSS_SELECTOR, "img[data-src], img[src]")
            
            for img in images[:num_images]:
                if self.collected_count >= self.max_images:
                    break
                
                img_url = img.get_attribute('data-src') or img.get_attribute('src')
                if not img_url or not img_url.startswith('http'):
                    continue
                
                filename = f"parking_officer_{self.collected_count:04d}.jpg"
                if self.download_image(img_url, filename):
                    self.collected_count += 1
                    collected += 1
                    print(f"収集: {filename} ({self.collected_count}/{self.max_images})")
                    time.sleep(0.5)  # レート制限対策
            
            driver.quit()
        except Exception as e:
            print(f"DuckDuckGo収集エラー: {e}")
        
        return collected
    
    def collect_from_unsplash(self, query: str, num_images: int = 10) -> int:
        """Unsplash APIから画像を収集（APIキー不要、制限あり）"""
        collected = 0
        try:
            # Unsplash Source API（APIキー不要、制限あり）
            url = f"https://source.unsplash.com/featured/?{quote(query)}"
            
            for i in range(num_images):
                if self.collected_count >= self.max_images:
                    break
                
                try:
                    response = requests.get(url, timeout=10, allow_redirects=True)
                    if response.status_code == 200:
                        filename = f"parking_officer_{self.collected_count:04d}.jpg"
                        filepath = self.output_dir / filename
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        
                        if filepath.stat().st_size > 1000:
                            self.collected_count += 1
                            collected += 1
                            print(f"収集: {filename} ({self.collected_count}/{self.max_images})")
                            time.sleep(1)
                except Exception as e:
                    print(f"Unsplash収集エラー: {e}")
                    continue
        except Exception as e:
            print(f"Unsplash収集エラー: {e}")
        
        return collected
    
    def collect_all(self) -> int:
        """すべての検索クエリから画像を収集"""
        total_collected = 0
        
        print(f"画像収集を開始します。最大 {self.max_images} 枚を収集します。")
        print(f"出力ディレクトリ: {self.output_dir}")
        print(f"\n注意: 画像収集には時間がかかります。")
        print(f"      手動で画像を dataset/images/ に配置することもできます。")
        
        # 既存の画像をカウント
        existing_images = len(list(self.output_dir.glob("*.jpg"))) + len(list(self.output_dir.glob("*.png")))
        if existing_images > 0:
            print(f"既存の画像: {existing_images} 枚")
            self.collected_count = existing_images
        
        for query in self.search_queries:
            if self.collected_count >= self.max_images:
                break
            
            print(f"\n検索クエリ: {query}")
            
            # DuckDuckGoから収集
            if SELENIUM_AVAILABLE:
                collected = self.collect_from_duckduckgo(query, num_images=15)
                total_collected += collected
            else:
                print("  Seleniumが利用できないため、このクエリをスキップします")
            
            if self.collected_count >= self.max_images:
                break
            
            time.sleep(2)  # レート制限対策
        
        print(f"\n収集完了: 合計 {total_collected} 枚の新しい画像を収集しました。")
        print(f"総画像数: {self.collected_count} 枚")
        
        if self.collected_count < self.min_images:
            print(f"\n警告: 収集画像数が少なすぎます ({self.collected_count} < {self.min_images})")
            print("      手動で画像を dataset/images/ に追加することをお勧めします。")
        
        return total_collected


def main():
    parser = argparse.ArgumentParser(description="駐車監視員の画像をネットから収集")
    parser.add_argument(
        "--output-dir",
        default="dataset/images",
        help="画像の出力ディレクトリ (default: dataset/images)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="収集する最大画像数 (default: 100)",
    )
    parser.add_argument(
        "--query",
        help="カスタム検索クエリ（指定しない場合はデフォルトクエリを使用）",
    )
    
    args = parser.parse_args()
    
    collector = ImageCollector(
        output_dir=args.output_dir,
        max_images=args.max_images,
    )
    
    if args.query:
        collector.search_queries = [args.query]
    
    collector.collect_all()


if __name__ == "__main__":
    main()

