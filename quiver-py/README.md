# quiver-py

Build locally with `maturin develop -m quiver-py/Cargo.toml`, then:

```python
from quiver_db import Index

index = Index("demo.qvdb", "demo.wal", 3)
vector_id = index.insert([1.0, 0.0, 0.0])
print(index.search([0.9, 0.1, 0.0], k=1))
index.delete(vector_id)
```
