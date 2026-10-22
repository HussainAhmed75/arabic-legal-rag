"""Cross-encoder reranker: BAAI/bge-reranker-v2-m3 (int8 ONNX). Reads (question, article) pairs together
and scores relevance — slower than retrieval, so it only re-orders the top candidates."""
import os

import numpy as np

MODEL, TOK = "models/rr_int8.onnx", "models/rr_tokenizer.json"


def available():
    return os.path.exists(MODEL) and os.path.exists(TOK)


class Reranker:
    def __init__(self, max_len=512):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(TOK)
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding(pad_id=1, pad_token="<pad>")

    def scores(self, query, docs, batch_size=8):
        out = []
        for i in range(0, len(docs), batch_size):
            enc = self.tok.encode_batch([(query, d) for d in docs[i:i + batch_size]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            out.append(self.sess.run(None, feed)[0].reshape(-1))
        return np.concatenate(out)
