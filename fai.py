import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -------- Load catalog CSV --------
df = pd.read_csv("catalog_enriched_1.csv")
df.columns = df.columns.str.lower()

# Make sure required columns exist
for col in ["assessment_name", "url", "description", "test_type", "skills"]:
    if col not in df.columns:
        raise ValueError(f"Missing column '{col}' in catalog_enriched_1.csv")


# -------- Test Type Classifier --------
def infer_test_type(text: str):
    text = text.lower()
    types = []
    if "personality" in text: types.append("Personality")
    if "motiv" in text: types.append("Motivation")
    if "situational" in text or "judgment" in text or "behaviour" in text or "behavior" in text:
        types.append("Behavioral")
        types.append("Judgment")
    if "cognitive" in text or "reasoning" in text or "logical" in text: types.append("Cognitive")
    if "skill" in text or "competenc" in text: types.append("Skill")
    if "simulation" in text: types.append("Simulation")
    if "coding" in text or "technical" in text or "program" in text: types.append("Technical")
    if "business" in text: types.append("Business")
    if not types: types.append("Other")
    return sorted(list(set(types)))


# -------- Skill Extractor --------
SKILL_LIST = ["teamwork", "communication", "problem solving", "reasoning", "coding",
              "python", "java", "leadership", "analysis", "decision making",
              "emotional intelligence", "collaboration", "management", "data",
              "logical", "business"]


def extract_skills(text: str):
    text = text.lower()
    found = []
    for skill in SKILL_LIST:
        ok = True
        for token in skill.split():
            if token not in text:
                ok = False
                break
        if ok:
            found.append(skill)
    return sorted(list(set(found)))


# -------- Load embedding model --------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------- Compute embeddings for descriptions --------
print("Computing embeddings...")
embeddings = np.array([model.encode(str(desc)).astype("float32") for desc in df["description"]])
print(f"Embeddings shape: {embeddings.shape}")

# -------- Build FAISS index --------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

# -------- Save index and metadata --------
faiss.write_index(index, "faiss_index.bin")

# Add skills and test_type to metadata
df["skills"] = df["description"].apply(extract_skills)
df["test_type"] = (df["name"].astype(str) + " " + df["description"].astype(str)).apply(infer_test_type)

with open("metadata.pkl", "wb") as f:
    pickle.dump(df, f)
print("FAISS index and metadata saved!")


# -------- Recommender function --------
def rec(query: str, top_k: int = 7):
    q_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, top_k)

    results = []
    for idx in indices[0]:
        row = df.iloc[idx]
        results.append({
            "name": row["name"],
            "url": row["url"],
            "description": row["description"],
            "test_type": row["test_type"],
            "skills": row["skills"]
        })
    return results


# -------- Quick test --------
if __name__ == "__main__":
    query = "Looking for a Python programmer with logical reasoning"
    from pprint import pprint

    pprint(rec(query))
