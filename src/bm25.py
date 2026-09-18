"""
Небольшая самодостаточная реализация Okapi BM25 (без внешних пакетов,
кроме numpy). Используется вместо стороннего пакета rank-bm25, чтобы
не создавать лишнюю точку отказа при офлайн-установке зависимостей.
"""
from collections import Counter
import math
import numpy as np

class BM25:
    def __init__(self, corpus_tokens, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.corpus_size else 0.0

        self.doc_freqs = []
        df = Counter()
        for doc in corpus_tokens:
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            for term in freqs.keys():
                df[term] += 1
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens):
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        q_freqs = Counter(query_tokens)
        for term, qf in q_freqs.items():
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, doc_freqs in enumerate(self.doc_freqs):
                f = doc_freqs.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores
