from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

import numpy as np

_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False


class EmbeddingBackend:
    name: str = "base"
    def fit(self, corpus: List[str]) -> None:
        return
    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class STEmbedding(EmbeddingBackend):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.name = f"sentence-transformers:{model_name}"
        self.model = SentenceTransformer(model_name)
    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")


class SimpleTfidfEmbedding(EmbeddingBackend):
    def __init__(self, max_features: int = 50000):
        self.name = "simple-tfidf"
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.fitted = False

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        return re.findall(r"\w+", s.lower())

    def fit(self, corpus: List[str]) -> None:
        df: Dict[str, int] = {}
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        items = sorted(df.items(), key=lambda x: x[1], reverse=True)
        if self.max_features and len(items) > self.max_features:
            items = items[: self.max_features]

        self.vocab = {t: i for i, (t, _) in enumerate(items)}
        N = max(1, len(corpus))
        idf_vals = np.zeros(len(self.vocab), dtype="float32")
        for t, i in self.vocab.items():
            dfi = df[t]
            idf_vals[i] = math.log((N + 1) / (dfi + 1)) + 1.0
        self.idf = idf_vals
        self.fitted = True

    def _vectorize(self, texts: List[str]) -> np.ndarray:
        if not self.fitted or self.idf is None:
            raise RuntimeError("SimpleTfidfEmbedding: call fit(corpus) before encode(texts).")
        V = len(self.vocab)
        if V == 0:
            return np.zeros((len(texts), 1), dtype="float32")
        mat = np.zeros((len(texts), V), dtype="float32")
        for row, txt in enumerate(texts):
            counts: Dict[int, int] = {}
            for tok in self._tokenize(txt):
                idx = self.vocab.get(tok)
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1
            if not counts:
                continue
            for idx, c in counts.items():
                mat[row, idx] = c * self.idf[idx]
            norm = np.linalg.norm(mat[row])
            if norm > 0:
                mat[row] /= norm
        return mat

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._vectorize(texts)


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    def safe_normalize(M: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return M / norms

    def is_normalized(M: np.ndarray) -> bool:
        norms = np.linalg.norm(M, axis=1)
        return np.allclose(norms, 1.0, atol=1e-3)

    A2 = A if is_normalized(A) else safe_normalize(A)
    B2 = B if is_normalized(B) else safe_normalize(B)
    sim = np.clip(A2 @ B2.T, 0.0, 1.0)
    return sim


def compute_similarity_matrix(backend: EmbeddingBackend, queries: List[str], index_texts: List[str]) -> np.ndarray:
    Q = backend.encode(queries)
    I = backend.encode(index_texts)
    Q = np.asarray(Q, dtype="float32")
    I = np.asarray(I, dtype="float32")
    return cosine_sim_matrix(Q, I)


def select_backend(index_texts: List[str], model_name: Optional[str] = None) -> EmbeddingBackend:
    if _ST_AVAILABLE:
        return STEmbedding(model_name or "intfloat/multilingual-e5-large")
    be = SimpleTfidfEmbedding()
    be.fit(index_texts)
    return be
