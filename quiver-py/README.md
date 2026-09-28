# quiver-py

Build locally with `maturin develop -m quiver-py/Cargo.toml`, then:

```python
from quiver_db import Index

index = Index("demo.qvdb", "demo.wal", 3)
vector_id = index.insert([1.0, 0.0, 0.0])
print(index.search([0.9, 0.1, 0.0], k=1))
index.delete(vector_id)
```

Vectors can carry key-value metadata, and searches can be restricted to
vectors whose metadata matches a filter:

```python
index.insert([1.0, 0.0, 0.0], metadata={"category": "science", "year": 2024})
index.insert([0.0, 1.0, 0.0], metadata={"category": "sports", "year": 2024})

# Equality filter.
print(index.search([1.0, 0.0, 0.0], k=10,
                   filter={"Eq": {"key": "category", "value": "science"}}))

# Conjunction of filters.
print(index.search([1.0, 0.0, 0.0], k=10, filter={"And": [
    {"Eq": {"key": "category", "value": "science"}},
    {"Eq": {"key": "year", "value": 2024}},
]}))
```

Metadata values may be booleans, integers, floats, or strings. Vectors
inserted without metadata never match a filter.

Batch-built quantized indexes are also available. They are L2-only with no
metadata, no deletes, and no online inserts — build in memory or load a
snapshot saved earlier:

```python
from quiver_db import Sq8IndexPy, IvfPqIndexPy

sq8 = Sq8IndexPy.build(vectors, metric="l2")
sq8.save("index.qvsq")
sq8 = Sq8IndexPy.load("index.qvsq")
print(sq8.search([0.9, 0.1, 0.0], k=1))

ivfpq = IvfPqIndexPy.build(vectors, nlist=1024, m=32, ksub=256)
ivfpq.save("index.qvpq")
ivfpq = IvfPqIndexPy.load("index.qvpq")
print(ivfpq.search([0.9, 0.1, 0.0], k=10, nprobe=8, rerank_factor=16))
```
