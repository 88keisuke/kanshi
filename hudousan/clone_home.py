import os
import requests
from bs4 import BeautifulSoup
import urllib.parse

BASE_URL = "https://uehon.jp"
START_URL = BASE_URL + "/"

SAVE_DIR = "home_clone"
IMG_DIR = os.path.join(SAVE_DIR, "images")
CSS_DIR = os.path.join(SAVE_DIR, "css")
JS_DIR  = os.path.join(SAVE_DIR, "js")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CSS_DIR, exist_ok=True)
os.makedirs(JS_DIR,  exist_ok=True)

def download_file(url, save_path):
    try:
        data = requests.get(url).content
        with open(save_path, "wb") as f:
            f.write(data)
        print(f"[SAVE] {save_path}")
    except Exception as e:
        print(f"[ERROR] downloading {url}: {e}")

def process_page(url):
    print(f"Cloning {url}")
    res = requests.get(url)
    res.encoding = res.apparent_encoding
    soup = BeautifulSoup(res.text, "html.parser")

    # 画像
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        full = urllib.parse.urljoin(BASE_URL, src)
        filename = os.path.basename(urllib.parse.urlparse(full).path)
        save_path = os.path.join(IMG_DIR, filename)
        if not os.path.exists(save_path):
            download_file(full, save_path)
        img["src"] = f"images/{filename}"

    # CSS
    for link in soup.find_all("link", {"rel": "stylesheet"}):
        href = link.get("href")
        if not href:
            continue
        full = urllib.parse.urljoin(BASE_URL, href)
        filename = os.path.basename(urllib.parse.urlparse(full).path)
        save_path = os.path.join(CSS_DIR, filename)
        if not os.path.exists(save_path):
            download_file(full, save_path)
        link["href"] = f"css/{filename}"

    # JS
    for script in soup.find_all("script", {"src": True}):
        src = script.get("src")
        full = urllib.parse.urljoin(BASE_URL, src)
        filename = os.path.basename(urllib.parse.urlparse(full).path)
        save_path = os.path.join(JS_DIR, filename)
        if not os.path.exists(save_path):
            download_file(full, save_path)
        script["src"] = f"js/{filename}"

    # HTML保存
    save_path = os.path.join(SAVE_DIR, "index.html")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    print(f"[DONE] Saved to {save_path}")

if __name__ == "__main__":
    process_page(START_URL)
