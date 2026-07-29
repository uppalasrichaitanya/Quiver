//! Small local Python API for creating, inserting into, and searching a Quiver index.

use pyo3::{exceptions::PyValueError, prelude::*};
use quiver_core::{distance::Metric, index::hnsw::{HnswConfig, HnswIndex}};

fn py_error(error: impl std::fmt::Display) -> PyErr { PyValueError::new_err(error.to_string()) }

#[pyclass]
struct Index { inner: HnswIndex }

#[pymethods]
impl Index {
    #[new]
    #[pyo3(signature = (data_path, wal_path, dimension, m=16, ef_construction=100))]
    fn new(data_path: String, wal_path: String, dimension: u32, m: usize, ef_construction: usize) -> PyResult<Self> {
        let config = HnswConfig::new(m).with_ef_construction(ef_construction);
        Ok(Self { inner: HnswIndex::create(data_path, wal_path, dimension, Metric::Cosine, config).map_err(py_error)? })
    }

    fn insert(&mut self, vector: Vec<f32>) -> PyResult<u64> { self.inner.insert(&vector).map_err(py_error) }

    #[pyo3(signature = (vector, k=10, ef_search=100))]
    fn search(&self, vector: Vec<f32>, k: usize, ef_search: usize) -> PyResult<Vec<(u64, f32)>> {
        Ok(self.inner.search(&vector, k, ef_search).map_err(py_error)?.into_iter().map(|hit| (hit.vector_id, hit.distance)).collect())
    }

    fn delete(&mut self, id: u64) -> PyResult<()> { self.inner.delete(id).map_err(py_error) }
}

#[pyfunction]
fn version() -> &'static str { env!("CARGO_PKG_VERSION") }

#[pymodule]
fn quiver_db(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
