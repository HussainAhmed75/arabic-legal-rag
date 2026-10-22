"""Offline ONNX embedders (multilingual-e5-small, bge-m3) — no torch / no internet needed at runtime.
Files come from scripts/download_data.py."""
import os

import numpy as np


class OnnxE5:
    doc_prefix, query_prefix, pooling, name = "passage: ", "query: ", "mean", "e5-small"

    def __init__(self, model="models/e5_model_quantized.onnx", tok="models/e5_tokenizer.json", max_len=512):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(tok)
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding(pad_id=1, pad_token="<pad>")

    def encode(self, texts, normalize_embeddings=True, batch_size=16, **_):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[i:i + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            v = h[:, 0] if self.pooling == "cls" else (h * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
            if normalize_embeddings:
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
            out.append(v.astype(np.float32))
        return np.vstack(out)


class OnnxBgeM3(OnnxE5):
    """BAAI/bge-m3 dense embeddings (CLS pooling, no prefixes) — strong multilingual/Arabic retriever."""
    doc_prefix, query_prefix, pooling, name = "", "", "cls", "bge-m3"

    def __init__(self, max_len=512):
        super().__init__("models/bge_m3_int8.onnx", "models/bge_m3_tokenizer.json", max_len)


def best_available():
    """Pick the strongest embedder whose files exist (override with EMB=e5)."""
    choice = os.getenv("EMB", "")
    if choice != "e5" and os.path.exists("models/bge_m3_int8.onnx"):
        return OnnxBgeM3()
    if os.path.exists("models/e5_model_quantized.onnx"):
        return OnnxE5()
    return None


def available():
    return os.path.exists("models/e5_model_quantized.onnx") or os.path.exists("models/bge_m3_int8.onnx")
