import os

# テスト時に使う出力先フォルダ
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/test_cache')
os.makedirs(OUTPUT_DIR, exist_ok=True)
