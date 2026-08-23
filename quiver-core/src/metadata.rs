//! Key-value metadata and filter predicates for vector search.
//!
//! Each vector may carry key-value metadata assigned at insert time
//! (e.g., `{"category": "science", "year": 2024}`), and searches may be
//! restricted to vectors whose metadata satisfies a [`Filter`] predicate.
//!
//! ## Design
//!
//! - [`MetaValue`] is a scalar: string, integer, float, or boolean.
//! - [`Metadata`] maps string keys to [`MetaValue`]s. It is backed by a
//!   [`BTreeMap`] so iteration and serialization order are deterministic.
//! - [`Filter`] is the query predicate. The initial (naive) scope is equality
//!   tests plus conjunction; `Or`, `In`, and range predicates are deferred.
//!
//! [`Filter::matches`] is total: a missing key never matches an `Eq`, and an
//! empty `And` matches everything.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A scalar metadata value attached to a vector.
///
/// Serializes untagged, so JSON scalars map directly:
/// `true` → [`Bool`](MetaValue::Bool), `1` → [`Int`](MetaValue::Int),
/// `1.5` → [`Float`](MetaValue::Float), `"x"` → [`Str`](MetaValue::Str).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetaValue {
    /// A boolean value.
    Bool(bool),
    /// A 64-bit signed integer.
    Int(i64),
    /// A 64-bit floating-point number.
    Float(f64),
    /// A UTF-8 string.
    Str(String),
}

impl From<bool> for MetaValue {
    fn from(value: bool) -> Self {
        MetaValue::Bool(value)
    }
}

impl From<i64> for MetaValue {
    fn from(value: i64) -> Self {
        MetaValue::Int(value)
    }
}

impl From<f64> for MetaValue {
    fn from(value: f64) -> Self {
        MetaValue::Float(value)
    }
}

impl From<String> for MetaValue {
    fn from(value: String) -> Self {
        MetaValue::Str(value)
    }
}

impl From<&str> for MetaValue {
    fn from(value: &str) -> Self {
        MetaValue::Str(value.to_owned())
    }
}

/// Key-value metadata attached to a vector.
///
/// A thin wrapper over a [`BTreeMap`] — chosen over a hash map so iteration
/// and serialization order are deterministic. Serializes as a plain JSON
/// object, e.g. `{"category": "science", "year": 2024}`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Metadata(BTreeMap<String, MetaValue>);

impl Metadata {
    /// Create an empty metadata map.
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Insert a key-value pair, returning the previous value for the key if any.
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        value: impl Into<MetaValue>,
    ) -> Option<MetaValue> {
        self.0.insert(key.into(), value.into())
    }

    /// Look up a value by key.
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.0.get(key)
    }

    /// Whether the given key is present.
    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    /// Number of key-value pairs.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the map has no entries.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterate over key-value pairs in sorted key order.
    pub fn iter(&self) -> std::collections::btree_map::Iter<'_, String, MetaValue> {
        self.0.iter()
    }
}

/// A predicate over vector metadata, used to restrict search results.
///
/// The initial scope is deliberately naive: equality tests and conjunction.
/// `Or`, `In`, and range predicates are deferred.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Filter {
    /// Matches when the metadata contains `key` with exactly `value`.
    ///
    /// A missing key never matches, and values of different types never
    /// match (`Int(1)` is not equal to `Float(1.0)`).
    Eq {
        /// The metadata key to test.
        key: String,
        /// The value the key must hold.
        value: MetaValue,
    },
    /// Matches when every contained filter matches.
    ///
    /// An empty list matches everything (vacuous truth).
    And(Vec<Filter>),
}

impl Filter {
    /// Evaluate the predicate against a metadata map.
    ///
    /// Total: never errors. Missing keys fail `Eq`, and an empty `And`
    /// holds vacuously.
    pub fn matches(&self, metadata: &Metadata) -> bool {
        match self {
            Filter::Eq { key, value } => metadata.get(key) == Some(value),
            Filter::And(filters) => filters.iter().all(|filter| filter.matches(metadata)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> Metadata {
        let mut md = Metadata::new();
        md.insert("category", "science");
        md.insert("year", 2024i64);
        md.insert("score", 0.5f64);
        md.insert("published", true);
        md
    }

    fn eq(key: &str, value: impl Into<MetaValue>) -> Filter {
        Filter::Eq {
            key: key.to_owned(),
            value: value.into(),
        }
    }

    #[test]
    fn test_eq_matches_on_equal_value() {
        let md = sample_metadata();
        assert!(eq("category", "science").matches(&md));
        assert!(eq("year", 2024i64).matches(&md));
        assert!(eq("score", 0.5f64).matches(&md));
        assert!(eq("published", true).matches(&md));
    }

    #[test]
    fn test_eq_rejects_different_value() {
        let md = sample_metadata();
        assert!(!eq("category", "sports").matches(&md));
        assert!(!eq("year", 1999i64).matches(&md));
        assert!(!eq("published", false).matches(&md));
    }

    #[test]
    fn test_eq_rejects_missing_key() {
        let md = sample_metadata();
        assert!(!eq("nonexistent", "science").matches(&md));
        assert!(!eq("nonexistent", 0i64).matches(&Metadata::new()));
    }

    #[test]
    fn test_eq_rejects_different_type() {
        let md = sample_metadata();
        // Int(2024) != Float(2024.0): types must agree, not just values.
        assert!(!eq("year", 2024.0f64).matches(&md));
        assert!(!eq("score", 1i64).matches(&md));
    }

    #[test]
    fn test_and_requires_all_conjuncts() {
        let md = sample_metadata();
        let filter = Filter::And(vec![
            eq("category", "science"),
            eq("year", 2024i64),
            eq("published", true),
        ]);
        assert!(filter.matches(&md));

        let filter = Filter::And(vec![
            eq("category", "science"),
            eq("year", 1999i64), // fails
        ]);
        assert!(!filter.matches(&md));
    }

    #[test]
    fn test_and_empty_matches_everything() {
        assert!(Filter::And(Vec::new()).matches(&Metadata::new()));
        assert!(Filter::And(Vec::new()).matches(&sample_metadata()));
    }

    #[test]
    fn test_nested_and() {
        let md = sample_metadata();
        let filter = Filter::And(vec![
            Filter::And(vec![eq("category", "science"), eq("year", 2024i64)]),
            Filter::And(vec![eq("published", true)]),
        ]);
        assert!(filter.matches(&md));

        let filter = Filter::And(vec![
            Filter::And(vec![eq("category", "science")]),
            Filter::And(vec![eq("year", 1999i64)]),
        ]);
        assert!(!filter.matches(&md));
    }

    #[test]
    fn test_matches_against_empty_metadata() {
        let md = Metadata::new();
        assert!(!eq("category", "science").matches(&md));
        assert!(Filter::And(vec![]).matches(&md));
    }

    #[test]
    fn test_metadata_insert_overwrites() {
        let mut md = Metadata::new();
        assert_eq!(md.insert("k", "v1"), None);
        assert_eq!(md.insert("k", "v2"), Some(MetaValue::Str("v1".into())));
        assert_eq!(md.get("k"), Some(&MetaValue::Str("v2".into())));
        assert_eq!(md.len(), 1);
    }

    #[test]
    fn test_metadata_deterministic_iteration_order() {
        let mut md = Metadata::new();
        md.insert("zebra", 1i64);
        md.insert("apple", 2i64);
        md.insert("mango", 3i64);
        let keys: Vec<&str> = md.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["apple", "mango", "zebra"]);
    }

    #[test]
    fn test_serde_metadata_from_json() {
        let md: Metadata = serde_json::from_str(
            r#"{"category":"science","year":2024,"score":0.5,"published":true}"#,
        )
        .unwrap();
        assert_eq!(md.len(), 4);
        assert_eq!(md.get("category"), Some(&MetaValue::Str("science".into())));
        assert_eq!(md.get("year"), Some(&MetaValue::Int(2024)));
        assert_eq!(md.get("score"), Some(&MetaValue::Float(0.5)));
        assert_eq!(md.get("published"), Some(&MetaValue::Bool(true)));
    }

    #[test]
    fn test_serde_metadata_roundtrip() {
        let md = sample_metadata();
        let json = serde_json::to_string(&md).unwrap();
        let parsed: Metadata = serde_json::from_str(&json).unwrap();
        assert_eq!(md, parsed);
    }

    #[test]
    fn test_serde_metadata_is_plain_object() {
        let mut md = Metadata::new();
        md.insert("k", "v");
        assert_eq!(serde_json::to_string(&md).unwrap(), r#"{"k":"v"}"#);
    }

    #[test]
    fn test_serde_filter_roundtrip() {
        let filter = Filter::And(vec![
            eq("category", "science"),
            Filter::And(vec![eq("year", 2024i64), eq("published", true)]),
        ]);
        let json = serde_json::to_string(&filter).unwrap();
        let parsed: Filter = serde_json::from_str(&json).unwrap();
        assert_eq!(filter, parsed);
    }

    #[test]
    fn test_serde_filter_from_json() {
        let filter: Filter = serde_json::from_str(
            r#"{"And":[{"Eq":{"key":"category","value":"science"}},{"Eq":{"key":"year","value":2024}}]}"#,
        )
        .unwrap();
        assert!(filter.matches(&sample_metadata()));
        assert!(!filter.matches(&Metadata::new()));
    }
}
