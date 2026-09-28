//! Small local Python API for creating, opening, inserting into, and
//! searching a Quiver index.

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyDict, PyList, PyType},
};
use quiver_core::{
    distance::Metric,
    index::hnsw::{HnswConfig, HnswIndex},
    index::ivfpq::{IvfPqConfig, IvfPqIndex},
    index::sq8::Sq8Index,
    metadata::{Filter, Metadata},
};

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Convert a Python value (None, bool, int, float, str, list, dict) into the
/// equivalent JSON value. Bool is checked before int because Python booleans
/// are a subclass of int.
fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if value.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(flag) = value.downcast::<PyBool>() {
        return Ok(serde_json::Value::Bool(flag.is_true()));
    }
    if let Ok(integer) = value.extract::<i64>() {
        return Ok(serde_json::Value::Number(integer.into()));
    }
    if let Ok(float) = value.extract::<f64>() {
        return serde_json::Number::from_f64(float)
            .map(serde_json::Value::Number)
            .ok_or_else(|| py_error("metadata/filter float must be finite"));
    }
    if let Ok(text) = value.extract::<String>() {
        return Ok(serde_json::Value::String(text));
    }
    if let Ok(list) = value.downcast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list {
            items.push(py_to_json(&item)?);
        }
        return Ok(serde_json::Value::Array(items));
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (key, item) in dict {
            let key: String = key
                .extract()
                .map_err(|_| py_error("metadata/filter keys must be strings"))?;
            map.insert(key, py_to_json(&item)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    Err(py_error(format!(
        "unsupported metadata/filter value type: {value}"
    )))
}

#[pyclass]
struct Index {
    inner: HnswIndex,
}

#[pymethods]
impl Index {
    #[new]
    #[pyo3(signature = (data_path, wal_path, dimension, m=16, ef_construction=100))]
    fn new(
        data_path: String,
        wal_path: String,
        dimension: u32,
        m: usize,
        ef_construction: usize,
    ) -> PyResult<Self> {
        // Refuse to run over an existing database: `create` truncates the data
        // file, WAL, and metadata snapshot, which would silently destroy a
        // previously stored index.
        if std::path::Path::new(&data_path).exists() || std::path::Path::new(&wal_path).exists() {
            return Err(py_error(format!(
                "database already exists at {data_path:?} (or WAL {wal_path:?}); \
                 use Index.open to keep it"
            )));
        }
        let config = HnswConfig::new(m).with_ef_construction(ef_construction);
        Ok(Self {
            inner: HnswIndex::create(data_path, wal_path, dimension, Metric::Cosine, config)
                .map_err(py_error)?,
        })
    }

    /// Open an existing index created by [`Index::new`].
    ///
    /// Replays the WAL for crash recovery and reuses the persisted graph
    /// topology snapshot when present.
    #[pyo3(signature = (data_path, wal_path, m=16, ef_construction=100))]
    #[classmethod]
    fn open(
        _cls: Bound<'_, PyType>,
        data_path: String,
        wal_path: String,
        m: usize,
        ef_construction: usize,
    ) -> PyResult<Self> {
        let config = HnswConfig::new(m).with_ef_construction(ef_construction);
        Ok(Self {
            inner: HnswIndex::open(data_path, wal_path, config).map_err(py_error)?,
        })
    }

    /// Insert a vector, optionally with key-value metadata.
    ///
    /// `metadata` must be a dict of string keys to scalar values
    /// (bool / int / float / str), e.g. `{"category": "science", "year": 2024}`.
    #[pyo3(signature = (vector, metadata=None))]
    fn insert(&mut self, vector: Vec<f32>, metadata: Option<Bound<'_, PyAny>>) -> PyResult<u64> {
        let id = match metadata {
            Some(metadata) => {
                let json = py_to_json(&metadata)?;
                let metadata: Metadata = serde_json::from_value(json).map_err(py_error)?;
                self.inner.insert_with_metadata(&vector, metadata)
            }
            None => self.inner.insert(&vector),
        };
        id.map_err(py_error)
    }

    /// Search for the `k` nearest neighbors, optionally restricted to vectors
    /// whose metadata matches `filter`.
    ///
    /// `filter` mirrors the JSON wire format, e.g.
    /// `{"Eq": {"key": "category", "value": "science"}}` or
    /// `{"And": [ ... ]}`.
    #[pyo3(signature = (vector, k=10, ef_search=100, filter=None))]
    fn search(
        &self,
        vector: Vec<f32>,
        k: usize,
        ef_search: usize,
        filter: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let hits = match filter {
            Some(filter) => {
                let json = py_to_json(&filter)?;
                let filter: Filter = serde_json::from_value(json).map_err(py_error)?;
                self.inner.search_filtered(&vector, k, ef_search, &filter)
            }
            None => self.inner.search(&vector, k, ef_search),
        };
        Ok(hits
            .map_err(py_error)?
            .into_iter()
            .map(|hit| (hit.vector_id, hit.distance))
            .collect())
    }

    fn delete(&mut self, id: u64) -> PyResult<()> {
        self.inner.delete(id).map_err(py_error)
    }

    /// Replace the metadata attached to a live vector.
    ///
    /// `metadata` must be a dict of string keys to scalar values, as in
    /// [`Index::insert`]. Unknown or deleted IDs raise `ValueError`.
    fn update_metadata(&mut self, id: u64, metadata: Bound<'_, PyAny>) -> PyResult<()> {
        let json = py_to_json(&metadata)?;
        let metadata: Metadata = serde_json::from_value(json).map_err(py_error)?;
        self.inner.update_metadata(id, metadata).map_err(py_error)
    }

    /// Remove the metadata attached to a live vector.
    fn clear_metadata(&mut self, id: u64) -> PyResult<()> {
        self.inner.clear_metadata(id).map_err(py_error)
    }
}

fn parse_metric(name: &str) -> PyResult<Metric> {
    match name.to_ascii_lowercase().as_str() {
        "l2" | "euclidean" => Ok(Metric::L2),
        "dot" | "dotproduct" | "dot_product" | "ip" => Ok(Metric::DotProduct),
        "cosine" | "cos" => Ok(Metric::Cosine),
        _ => Err(py_error("metric must be one of 'l2', 'dot', 'cosine'")),
    }
}

/// Batch-built SQ8 flat index: no deletes, no metadata. Built in memory,
/// searchable directly, or persisted with `save`/`load`.
#[pyclass]
struct Sq8IndexPy {
    inner: Sq8Index,
}

#[pymethods]
impl Sq8IndexPy {
    #[staticmethod]
    #[pyo3(signature = (vectors, metric="l2"))]
    fn build(vectors: Vec<Vec<f32>>, metric: &str) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        Ok(Self {
            inner: Sq8Index::build(&vectors, metric).map_err(py_error)?,
        })
    }

    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        Ok(Self {
            inner: Sq8Index::load(&path).map_err(py_error)?,
        })
    }

    fn save(&self, path: String) -> PyResult<()> {
        self.inner.save(&path).map_err(py_error)
    }

    #[pyo3(signature = (vector, k=10))]
    fn search(&self, vector: Vec<f32>, k: usize) -> PyResult<Vec<(u64, f32)>> {
        Ok(self
            .inner
            .search(&vector, k)
            .map_err(py_error)?
            .into_iter()
            .map(|hit| (hit.vector_id, hit.distance))
            .collect())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

/// Batch-built IVF-PQ index (L2-only, no metadata). Built in memory,
/// searchable directly, or persisted with `save`/`load`.
#[pyclass]
struct IvfPqIndexPy {
    inner: IvfPqIndex,
}

#[pymethods]
impl IvfPqIndexPy {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (vectors, nlist, m, ksub, kmeans_iters=8, training_size=65536, store_vectors=true, seed=0xC0FFEE))]
    fn build(
        vectors: Vec<Vec<f32>>,
        nlist: usize,
        m: usize,
        ksub: usize,
        kmeans_iters: usize,
        training_size: usize,
        store_vectors: bool,
        seed: u64,
    ) -> PyResult<Self> {
        let mut config = IvfPqConfig::new(nlist, m, ksub);
        config.kmeans_iters = kmeans_iters;
        config.training_size = training_size;
        config.store_vectors = store_vectors;
        config.seed = seed;
        Ok(Self {
            inner: IvfPqIndex::build(&vectors, &config).map_err(py_error)?,
        })
    }

    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        Ok(Self {
            inner: IvfPqIndex::load(&path).map_err(py_error)?,
        })
    }

    fn save(&self, path: String) -> PyResult<()> {
        self.inner.save(&path).map_err(py_error)
    }

    #[pyo3(signature = (vector, k=10, nprobe=8, rerank_factor=0))]
    fn search(
        &self,
        vector: Vec<f32>,
        k: usize,
        nprobe: usize,
        rerank_factor: usize,
    ) -> PyResult<Vec<(u64, f32)>> {
        Ok(self
            .inner
            .search(&vector, k, nprobe, rerank_factor)
            .map_err(py_error)?
            .into_iter()
            .map(|hit| (hit.vector_id, hit.distance))
            .collect())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn quiver_db(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    m.add_class::<Sq8IndexPy>()?;
    m.add_class::<IvfPqIndexPy>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
