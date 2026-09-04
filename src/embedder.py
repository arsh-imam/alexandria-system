"""
AlignedEmbedder — query-time embedder that EXACTLY replicates the index
build path (embed_gpu.py on the Nitro): float32 onnx/model.onnx,
fixed 128-token padding + truncation, plain mean over all 128 positions
(NOT masked mean — the build didn't mask, so we must not either),
then L2 normalize. Any deviation puts queries in a different space
than the 2,028,337 index vectors.
"""
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

EMBED_DIR      = "/media/pi/KINGSTON/local_ai/models/embed"
MODEL_ONNX     = os.path.join(EMBED_DIR, "model.onnx")
TOKENIZER_JSON = os.path.join(EMBED_DIR, "tokenizer.json")


class AlignedEmbedder:

    def __init__(self, threads=4):
        if not os.path.exists(MODEL_ONNX):
            raise FileNotFoundError(
                f"{MODEL_ONNX} missing — run setup_phase1.py first")
        self.tok = Tokenizer.from_file(TOKENIZER_JSON)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
        self.tok.enable_truncation(max_length=128)

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            MODEL_ONNX, so, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def embed(self, texts):
        enc  = self.tok.encode_batch(list(texts))
        ids  = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        out   = self.sess.run(None, feeds)
        embs  = out[0].mean(axis=1)          # plain mean, matches build
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return (embs / np.maximum(norms, 1e-9)).astype(np.float32)

    def embed_one(self, text):
        return self.embed([text])[0]
