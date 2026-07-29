"""End-to-end semantic search against quiver-server.

Uses scikit-learn's off-the-shelf HashingVectorizer and needs no model download.
Run the server with QUIVER_DIMENSION=384, then run this script.
"""
import json
import urllib.request

from sklearn.feature_extraction.text import HashingVectorizer

CORPUS = [
    "The Moon is Earth's only natural satellite and orbits at about 384,400 kilometres.",
    "Rust is a systems programming language focused on safety, speed, and concurrency.",
    "The Bengal tiger is a population of the Panthera tigris subspecies native to South Asia.",
    "The Python programming language emphasizes code readability and a concise syntax.",
    "Photosynthesis converts light energy into chemical energy in plants and algae.",
]


def request(method, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:8080{path}", body, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as response:
        data = response.read()
        return json.loads(data) if data else None


model = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2")
vectors = model.transform(CORPUS).astype("float32").toarray()
documents = {}
for text, vector in zip(CORPUS, vectors, strict=True):
    result = request("POST", "/vectors", {"vector": vector.tolist()})
    documents[result["id"]] = text

query = "Which language is designed for safe low-level software?"
query_vector = model.transform([query]).astype("float32").toarray()[0].tolist()
hits = request("POST", "/search", {"vector": query_vector, "k": 3, "ef_search": 50})
print(f"query: {query}\n")
for rank, hit in enumerate(hits, 1):
    print(f"{rank}. {documents[hit['id']]}  (distance={hit['distance']:.4f})")

# The delete endpoint is part of the same minimal API.
request("DELETE", f"/vectors/{hits[-1]['id']}")
