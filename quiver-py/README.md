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
