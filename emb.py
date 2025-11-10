# precompute_emb.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("catalog_enriched_1.csv")
df.columns = df.columns.str.lower()

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
texts = df["description"].astype(str).tolist()

# encode all descriptions
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

# save numpy array
np.save("embeddings.npy", embeddings)
print("Saved embeddings.npy")
