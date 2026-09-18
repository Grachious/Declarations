"""
Опциональный семантический сигнал на основе эмбеддингов.
  1) Заранее выполняется `python download_model.py`,
     который сохраняет sentence-transformers модель в ./models/embedder.
  2) `run.py` проверяет наличие ./models/embedder. Если модель на месте, то
     она загружается только с дискаи используется как дополнительный семантический сигнал. 
     Если модели нет, то модуль отключается и решение работает на лексических сигналах (BM25 + TF-IDF).

Модель по умолчанию: intfloat/multilingual-e5-base (~278M параметров.
"""
import os
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "embedder")


def embeddings_available() -> bool:
    return os.path.isdir(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0


def _load_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_DIR, device="cpu")
    return model


def compute_embedding_similarity(dec_texts, reg_texts, batch_size: int = 32):
    if not embeddings_available():
        return None
    try:
        model = _load_model()
    except Exception as e:
        print(f"[embeddings] модель найдена, но не удалось загрузить ({e}); "
              f"пропускаем семантический сигнал.")
        return None
    dec_inputs = [f"query: {t}" for t in dec_texts]
    reg_inputs = [f"passage: {t}" for t in reg_texts]

    dec_emb = model.encode(dec_inputs, batch_size=batch_size, show_progress_bar=False,
                            normalize_embeddings=True, convert_to_numpy=True)
    reg_emb = model.encode(reg_inputs, batch_size=batch_size, show_progress_bar=False,
                            normalize_embeddings=True, convert_to_numpy=True)
    sim = dec_emb @ reg_emb.T
    return sim.astype(np.float64)
