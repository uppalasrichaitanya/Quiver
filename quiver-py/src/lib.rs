//! Small local Python API for creating, inserting into, and searching a Quiver index.

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyDict, PyList},
};
use quiver_core::{
    distance::Metric,
    index::hnsw::{HnswConfig, HnswIndex},
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
        let config = HnswConfig::new(m).with_ef_construction(ef_construction);
        Ok(Self {
            inner: HnswIndex::create(data_path, wal_path, dimension, Metric::Cosine, config)
                .map_err(py_error)?,
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
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn quiver_db(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
