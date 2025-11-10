from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rec import recommend  # import function from rec.py

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str
    top_k: int = 3

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend_api(q: Query):

    results = recommend(q.query, q.top_k)
    print("Returned results from recommend():", results)
    return {"assessments": results}
