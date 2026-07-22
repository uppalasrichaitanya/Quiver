//! Python bindings for the Quiver vector search engine.
//!
//! This module exposes the core engine to Python via PyO3/maturin,
//! installable as `pip install quiver-db`.
//!
//! Full bindings will be implemented in Phase 10.

use pyo3::prelude::*;

/// Returns the version of the quiver-db library.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python module definition for quiver_db.
#[pymodule]
fn quiver_db(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
