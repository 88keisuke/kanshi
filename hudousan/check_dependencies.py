# -*- coding: utf-8 -*-
"""
依存パッケージのチェックと自動インストール

このスクリプトは必要なパッケージがインストールされているか確認し、
不足している場合はインストールを試みます。
"""

import sys
import subprocess
from pathlib import Path


REQUIRED_PACKAGES = {
    'ultralytics': 'ultralytics',
    'cv2': 'opencv-python',
    'requests': 'requests',
    'bs4': 'beautifulsoup4',
    'selenium': 'selenium',
    'yaml': 'pyyaml',
}

OPTIONAL_PACKAGES = {
    'selenium': 'selenium',  # 画像収集で使用（オプション）
}


def check_package(package_name: str, import_name: str = None) -> bool:
    """パッケージがインストールされているかチェック"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def install_package(package_name: str) -> bool:
    """パッケージをインストール"""
    try:
        print(f"  インストール中: {package_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False


def check_and_install(required: bool = True, auto_install: bool = False):
    """依存パッケージをチェックし、必要に応じてインストール"""
    missing_packages = []
    optional_missing = []
    
    print("依存パッケージをチェックしています...")
    print()
    
    # 必須パッケージのチェック
    for import_name, package_name in REQUIRED_PACKAGES.items():
        if check_package(package_name, import_name):
            print(f"✓ {package_name}")
        else:
            print(f"✗ {package_name} (未インストール)")
            if required:
                missing_packages.append(package_name)
            else:
                optional_missing.append(package_name)
    
    # オプションパッケージのチェック
    for import_name, package_name in OPTIONAL_PACKAGES.items():
        if package_name not in REQUIRED_PACKAGES.values():
            if check_package(package_name, import_name):
                print(f"✓ {package_name} (オプション)")
            else:
                print(f"○ {package_name} (オプション、未インストール)")
                optional_missing.append(package_name)
    
    print()
    
    # 不足しているパッケージがある場合
    if missing_packages:
        if auto_install:
            print("不足しているパッケージをインストールします...")
            print()
            failed = []
            for package in missing_packages:
                if not install_package(package):
                    failed.append(package)
            
            if failed:
                print()
                print("エラー: 以下のパッケージのインストールに失敗しました:")
                for package in failed:
                    print(f"  - {package}")
                print()
                print("手動でインストールしてください:")
                print(f"  pip install {' '.join(failed)}")
                return False
            else:
                print()
                print("✓ すべてのパッケージがインストールされました")
                return True
        else:
            print("エラー: 以下のパッケージがインストールされていません:")
            for package in missing_packages:
                print(f"  - {package}")
            print()
            print("インストール方法:")
            print(f"  pip install {' '.join(missing_packages)}")
            print()
            print("または、自動インストールスクリプトを実行:")
            print("  ./install_dependencies.sh")
            return False
    
    if optional_missing:
        print("注意: 以下のオプションパッケージがインストールされていません:")
        for package in optional_missing:
            print(f"  - {package}")
        print("（機能が制限される可能性があります）")
        print()
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="依存パッケージのチェック")
    parser.add_argument(
        "--auto-install",
        action="store_true",
        help="不足しているパッケージを自動的にインストール",
    )
    parser.add_argument(
        "--optional",
        action="store_true",
        help="オプションパッケージも必須として扱う",
    )
    
    args = parser.parse_args()
    
    success = check_and_install(
        required=not args.optional,
        auto_install=args.auto_install
    )
    
    sys.exit(0 if success else 1)

