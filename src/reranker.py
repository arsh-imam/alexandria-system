"""
Cross-encoder reranker (ms-marco-MiniLM-L-6-v2, INT8 ONNX).
Reads the query and each candidate passage TOGETHER through
self-attention and outputs a relevance logit. This is the
arbiter of the whole retrieval pipeline: candidates from all
generators are scored here, and the final confidence signal
is derived from these logits.
Measured on Pi 5: ~50 ms per (query, passage) pair.
"""
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

RERANK_DIR = "/media/pi/KINGSTON/local_ai/models/rerank"


class Reranker:

    def __init__(self, threads=4):
        self.tok = Tokenizer.from_file(
            os.path.join(RERANK_DIR, "tokenizer.json"))
        self.tok.enable_truncation(max_length=256)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            os.path.join(RERANK_DIR, "model_quantized.onnx"),
            so, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def score(self, query, passages, batch=8):
        """Return one relevance logit per passage. Higher = better."""
        scores = []
        for b in range(0, len(passages), batch):
            chunk = passages[b:b + batch]
            enc = self.tok.encode_batch([(query, p) for p in chunk])
            ids  = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.array(
                    [e.type_ids for e in enc], dtype=np.int64)
            out = self.sess.run(None, feeds)[0]
            scores.extend(out.reshape(-1).tolist())
        return scores
