"""Download the law text (Arabic Wikisource) and the ONNX models (Hugging Face), then build data/articles.jsonl.
Usage: python scripts/download_data.py            # law + bge-m3 + reranker
       python scripts/download_data.py --small    # law + multilingual-e5-small only (~120 MB)
"""
import os
import subprocess
import sys
import urllib.parse
import urllib.request

LAW_TITLE = "قانون_العمل_14_لسنة_2025_-_مصر"
LAW_URL = "https://ar.wikisource.org/w/index.php?title=" + urllib.parse.quote(LAW_TITLE) + "&action=raw"
HF = "https://huggingface.co/{repo}/resolve/main/{path}"
MODELS = {
    "small": [("Xenova/multilingual-e5-small", "onnx/model_quantized.onnx", "e5_model_quantized.onnx"),
              ("Xenova/multilingual-e5-small", "tokenizer.json", "e5_tokenizer.json")],
    "full": [("Xenova/bge-m3", "onnx/model_int8.onnx", "bge_m3_int8.onnx"),
             ("Xenova/bge-m3", "tokenizer.json", "bge_m3_tokenizer.json"),
             ("onnx-community/bge-reranker-v2-m3-ONNX", "onnx/model_int8.onnx", "rr_int8.onnx"),
             ("onnx-community/bge-reranker-v2-m3-ONNX", "tokenizer.json", "rr_tokenizer.json")],
}


def fetch(url, dest):
    if os.path.exists(dest):
        print("exists:", dest)
        return
    print("downloading", dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def main():
    fetch(LAW_URL, "data/raw/labour_law_14_2025.wikitext")
    subprocess.run([sys.executable, "-m", "src.parse_law", "data/raw/labour_law_14_2025.wikitext",
                    "--law", "قانون العمل 14 لسنة 2025", "--out", "data/articles.jsonl"], check=True)
    for repo, path, name in MODELS["small" if "--small" in sys.argv else "full"]:
        fetch(HF.format(repo=repo, path=path), f"models/{name}")


if __name__ == "__main__":
    main()
