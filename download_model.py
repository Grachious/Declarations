import os

MODEL_NAME = "intfloat/multilingual-e5-base"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "embedder")

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    os.makedirs(OUT_DIR, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    model.save(OUT_DIR)
