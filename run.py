"""
Алгоритм:
  1. Собираем текст декларации (G31_1 + desc_extention) и текст НПА (npa).
  2. Нормализуем оба текста.
  3. Считаем два лексических сигнала: BM25 и TF-IDF cosine.
  4. Если заранее была скачана модель эмбеддингов (models/embedder), то
     добавляем семантический сигнал.
  5. Компонуем сигналы.
  6. Для каждой декларации берём top-10 НПА по убыванию итогового скора.
  7. Формируем predictions.csv.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from text_norm import tokenize, normalize_text 
from bm25 import BM25 
from embeddings import compute_embedding_similarity, embeddings_available 


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_declaration_text(d: dict) -> str:
    parts = [d.get("G31_1") or ""]
    ext = (d.get("desc_extention") or "").strip()
    if len(ext.strip(" :;,.-")) > 0:
        parts.append(ext)
    return " ".join(p for p in parts if p)


def minmax_norm(mat: np.ndarray) -> np.ndarray:
    mn = mat.min(axis=1, keepdims=True)
    mx = mat.max(axis=1, keepdims=True)
    rng = mx - mn
    out = np.zeros_like(mat)
    nonzero = rng[:, 0] > 1e-12
    out[nonzero] = (mat[nonzero] - mn[nonzero]) / rng[nonzero]
    return out


def main():
    ap = argparse.ArgumentParser(description="Ранжирование НПА по релевантности к таможенным декларациям.")
    ap.add_argument("--declarations", default=os.path.join("data", "declarations.jsonl"))
    ap.add_argument("--regulations", default=os.path.join("data", "regulations.jsonl"))
    ap.add_argument("--out", default="./out")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--w-bm25", type=float, default=0.5)
    ap.add_argument("--w-tfidf", type=float, default=0.2)
    ap.add_argument("--w-embed", type=float, default=0.3)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    decs = read_jsonl(args.declarations)
    regs = read_jsonl(args.regulations)
    print(f"  деклараций: {len(decs)}, регуляций: {len(regs)}")

    if len(regs) < args.top_k:
        raise ValueError(f"В справочнике регуляций ({len(regs)}) меньше, чем top-k={args.top_k}")

    dec_ids = [d["declaration_id"] for d in decs]
    reg_ids = [r["regulation_id"] for r in regs]

    dec_raw_texts = [build_declaration_text(d) for d in decs]
    reg_raw_texts = [r.get("npa") or "" for r in regs]
    dec_tokens = [tokenize(t) for t in dec_raw_texts]
    reg_tokens = [tokenize(t) for t in reg_raw_texts]
    dec_norm = [" ".join(toks) for toks in dec_tokens]
    reg_norm = [" ".join(toks) for toks in reg_tokens]

    print("BM25")
    bm25 = BM25(reg_tokens)
    bm25_scores = np.zeros((len(decs), len(regs)), dtype=np.float64)
    for i, q in enumerate(dec_tokens):
        bm25_scores[i] = bm25.get_scores(q)

    print("TF-IDF cosine")
    vectorizer = TfidfVectorizer(
        analyzer=str.split, token_pattern=None, sublinear_tf=True, min_df=1
    )
    vectorizer.fit(dec_norm + reg_norm)
    dec_tfidf = vectorizer.transform(dec_norm)
    reg_tfidf = vectorizer.transform(reg_norm)
    tfidf_scores = cosine_similarity(dec_tfidf, reg_tfidf)

    embed_scores = None
    if embeddings_available():
        print("Найдена локальная модель эмбеддингов")
        embed_scores = compute_embedding_similarity(dec_raw_texts, reg_raw_texts)
    else:
        print("Используются только лексические сигналы.")

    bm25_n = minmax_norm(bm25_scores)
    tfidf_n = minmax_norm(tfidf_scores)

    if embed_scores is not None:
        embed_n = minmax_norm(embed_scores)
        w_sum = args.w_bm25 + args.w_tfidf + args.w_embed
        w_bm25, w_tfidf, w_embed = args.w_bm25 / w_sum, args.w_tfidf / w_sum, args.w_embed / w_sum
        final = w_bm25 * bm25_n + w_tfidf * tfidf_n + w_embed * embed_n
    else:
        w_sum = args.w_bm25 + args.w_tfidf
        w_bm25, w_tfidf = args.w_bm25 / w_sum, args.w_tfidf / w_sum
        final = w_bm25 * bm25_n + w_tfidf * tfidf_n

    print("Формирование top-k")
    rows = []
    reg_ids_arr = np.array(reg_ids)
    for i, decl_id in enumerate(dec_ids):
        scores_i = final[i]
        order = sorted(range(len(regs)), key=lambda j: (-scores_i[j], reg_ids_arr[j]))
        top = order[: args.top_k]
        for rank, j in enumerate(top, start=1):
            rows.append({
                "declaration_id": decl_id,
                "rank": rank,
                "regulation_id": reg_ids_arr[j],
                "score": float(scores_i[j]),
            })

    out_df = pd.DataFrame(rows, columns=["declaration_id", "rank", "regulation_id", "score"])
    out_path = os.path.join(args.out, "predictions.csv")
    out_df.to_csv(out_path, index=False)
    _validate(out_df, dec_ids, args.top_k)

    elapsed = time.time() - t0
    print(f"Готово: {out_path} ({len(out_df)} строк). Время выполнения: {elapsed:.1f} c.")


def _validate(df: pd.DataFrame, dec_ids, top_k: int):
    errors = []
    dec_ids_set = set(dec_ids)
    grouped = df.groupby("declaration_id")
    if set(grouped.groups.keys()) != dec_ids_set:
        errors.append("Не все декларации присутствуют в выходном файле (или есть лишние).")
    for decl_id, g in grouped:
        if len(g) != top_k:
            errors.append(f"{decl_id}: {len(g)} строк вместо {top_k}")
        if g["regulation_id"].nunique() != top_k:
            errors.append(f"{decl_id}: повторяющиеся regulation_id")
        if sorted(g["rank"].tolist()) != list(range(1, top_k + 1)):
            errors.append(f"{decl_id}: некорректные ранги")
        if not np.isfinite(g["score"]).all():
            errors.append(f"{decl_id}: нечисловой score")
    if errors:
        print("ОШИБКИ ФОРМАТА:")
        for e in errors[:20]:
            print("  -", e)
        raise SystemExit(1)
    print("Самопроверка формата пройдена")


if __name__ == "__main__":
    main()
