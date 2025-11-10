# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
#
# # -------- Load catalog CSV --------
# df = pd.read_csv("catalog_enriched_1.csv")
# df.columns = df.columns.str.lower()
#
# # Ensure required columns exist
# REQUIRED = ["assessment_name", "url", "description", "test_type", "skills"]
# for col in REQUIRED:
#     if col not in df.columns:
#         raise ValueError(f"Missing column {col} in catalog_enriched_1.csv")
#
# # -------- Embedding Model --------
# _model = None
#
# def get_model():
#     global _model
#     if _model is None:
#         from sentence_transformers import SentenceTransformer
#         _model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
#     return _model
#
# model = get_model()
# #---- Test Type Classifier --------
# emb_matrix = np.load("embeddings.npy")
# def infer_test_type(text: str):
#     text = text.lower()
#     types = []
#
#     if "personality" in text or "opq" in text:
#         types.append("Personality")
#     if "motiv" in text or "mq" in text:
#         types.append("Motivation")
#     if "situational" in text or "judgment" in text or "behaviour" in text or "behavior" in text or "sjt" in text:
#         types.append("Behavioral")
#         types.append("Judgment")
#     if "cognitive" in text or "reasoning" in text or "logical" in text:
#         types.append("Cognitive")
#     if "skill" in text or "competenc" in text:
#         types.append("Skill")
#     if "simulation" in text:
#         types.append("Simulation")
#     if "coding" in text or "technical" in text or "program" in text:
#         types.append("Technical")
#     if "business" in text:
#         types.append("Business")
#     if not types:
#         types.append("Other")
#
#     return sorted(list(set(types)))
#
# # -------- Skill Extractor --------
# SKILL_LIST = [
#     "teamwork","communication","problem solving","reasoning","coding","python","java",
#     "leadership","analysis","decision making","emotional intelligence","collaboration",
#     "management", "data", "logical", "business"
# ]
#
# def extract_skills(text: str):
#     text = text.lower()
#     found = []
#
#     for skill in SKILL_LIST:
#         ok = True
#         for token in skill.split():
#             if token not in text:
#                 ok = False
#                 break
#         if ok:
#             found.append(skill)
#
#     return sorted(list(set(found)))
#
# # -------- RECOMMENDER FUNCTION --------
# def recommend(query: str, top_k: int = 7):
#     q_emb = model.encode(query)
#     scores = util.cos_sim(q_emb, emb_matrix).squeeze()
#     top_idx = scores.topk(top_k).indices.tolist()
#
#     results = []
#     for i in top_idx:
#         row = df.iloc[i]
#         text = f"{row['assessment_name']} {row['description']}"
#         results.append({
#             "name": row["assessment_name"],  # renamed for API
#             "url": row["url"],
#             "description": row["description"] or "No description",
#             "test_type": infer_test_type(f"{row['assessment_name']} {row['description']}") or ["Other"],
#             "skills": extract_skills(row["description"]) or ["Core Competency"]
#         })
#
#     return results
#
# # -------- Quick test --------
# if __name__ == "__main__":
#     query = "Looking for a Python programmer with logical reasoning"
#     from pprint import pprint
#     pprint(recommend(query))
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ---------------- LOAD CATALOG ----------------
df = pd.read_csv("catalog_enriched_1.csv")
df.columns = df.columns.str.lower()

REQUIRED = ["assessment_name", "url", "description", "test_type", "skills"]
for col in REQUIRED:
    if col not in df.columns:
        raise ValueError(f"Missing column {col} in catalog_enriched_1.csv")


# ---------------- LOAD MODEL ----------------
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return _model

model = get_model()


# ---------------- INFER TEST TYPE -------------
def infer_test_type(text: str):
    text = text.lower()
    types = []

    if "personality" in text or "opq" in text:
        types.append("Personality")
    if "motiv" in text or "mq" in text:
        types.append("Motivation")
    if "situational" in text or "judgment" in text or "behaviour" in text or "behavior" in text or "sjt" in text:
        types.append("Behavioral")
        types.append("Judgment")
    if "cognitive" in text or "reasoning" in text or "logical" in text:
        types.append("Cognitive")
    if "skill" in text or "competenc" in text:
        types.append("Skill")
    if "simulation" in text:
        types.append("Simulation")
    if "coding" in text or "technical" in text or "program" in text:
        types.append("Technical")
    if "business" in text:
        types.append("Business")
    if not types:
        types.append("Other")

    return sorted(list(set(types)))


# ---------------- SKILL EXTRACTION ----------------
SKILL_LIST = [
    "teamwork", "communication", "problem solving", "reasoning",
    "coding", "python", "java", "leadership", "analysis",
    "decision making", "emotional intelligence","collaboration",
    "management", "data", "logical", "business"
]

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


# ---------------- RECOMMENDER (memory-safe) ----------------
def recommend(query: str, top_k: int = 7):

    # Encode query
    q_emb = model.encode(query)

    desc_list = df["description"].astype(str).tolist()
    scores = []

    # BATCHED ENCODING — LOW MEMORY
    batch_size = 32
    for i in range(0, len(desc_list), batch_size):
        batch = desc_list[i:i+batch_size]
        batch_emb = model.encode(batch)
        sc = util.cos_sim(q_emb, batch_emb)
        scores.extend(sc.squeeze().tolist())

    # pick top score indices
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in idx:
        row = df.iloc[i]
        results.append({
            "name": row["assessment_name"],
            "url": row["url"],
            "description": row["description"] or "No description",
            "test_type": infer_test_type(f"{row['assessment_name']} {row['description']}") or ["Other"],
            "skills": extract_skills(row["description"]) or ["Core Competency"]
        })

    return results


# ---------------- QUICK TEST ----------------
if __name__ == "__main__":
    from pprint import pprint
    q = "Looking for a Python programmer with logical reasoning"
    pprint(recommend(q))
