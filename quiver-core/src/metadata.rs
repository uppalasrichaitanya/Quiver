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
use std::io::{Cursor, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};

use crate::error::{QuiverError, Result};

/// Current version of the binary metadata encoding (see [`Metadata::to_bytes`]).
pub const METADATA_ENCODING_VERSION: u8 = 1;

const TAG_BOOL: u8 = 0;
const TAG_INT: u8 = 1;
const TAG_FLOAT: u8 = 2;
const TAG_STR: u8 = 3;

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

    /// Serialize to the versioned binary encoding used by the WAL and the
    /// metadata snapshot.
    ///
    /// ```text
    /// [u8 version][u32 entry_count]
    /// per entry (sorted by key):
    ///   [u32 key_len][key UTF-8][u8 value tag][value payload]
    ///   tags: 0 = bool [u8], 1 = int [i64 LE], 2 = float [f64 LE],
    ///         3 = str [u32 len][bytes]
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u8(METADATA_ENCODING_VERSION).unwrap();
        buf.write_u32::<LittleEndian>(self.0.len() as u32).unwrap();
        for (key, value) in &self.0 {
            buf.write_u32::<LittleEndian>(key.len() as u32).unwrap();
            buf.write_all(key.as_bytes()).unwrap();
            match value {
                MetaValue::Bool(b) => {
                    buf.write_u8(TAG_BOOL).unwrap();
                    buf.write_u8(u8::from(*b)).unwrap();
                }
                MetaValue::Int(i) => {
                    buf.write_u8(TAG_INT).unwrap();
                    buf.write_i64::<LittleEndian>(*i).unwrap();
                }
                MetaValue::Float(f) => {
                    buf.write_u8(TAG_FLOAT).unwrap();
                    buf.write_f64::<LittleEndian>(*f).unwrap();
                }
                MetaValue::Str(s) => {
                    buf.write_u8(TAG_STR).unwrap();
                    buf.write_u32::<LittleEndian>(s.len() as u32).unwrap();
                    buf.write_all(s.as_bytes()).unwrap();
                }
            }
        }
        buf
    }

    /// Parse the binary encoding produced by [`Metadata::to_bytes`].
    ///
    /// Bounds-checked throughout: malformed input returns an error rather than
    /// panicking or allocating unbounded memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let invalid = |message: String| QuiverError::InvalidFormat(message);

        let mut cursor = Cursor::new(bytes);
        let version = cursor
            .read_u8()
            .map_err(|_| invalid("metadata blob too short".to_string()))?;
        if version != METADATA_ENCODING_VERSION {
            return Err(invalid(format!(
                "unsupported metadata encoding version: {version}"
            )));
        }
        let entry_count = cursor
            .read_u32::<LittleEndian>()
            .map_err(|_| invalid("metadata blob missing entry count".to_string()))?;

        let mut map = BTreeMap::new();
        for _ in 0..entry_count {
            let key_len = cursor
                .read_u32::<LittleEndian>()
                .ok()
                .and_then(|len| usize::try_from(len).ok())
                .ok_or_else(|| invalid("metadata key length invalid".to_string()))?;
            let start = usize::try_from(cursor.position())
                .ok()
                .and_then(|pos| pos.checked_add(key_len))
                .ok_or_else(|| invalid("metadata key length overflow".to_string()))?;
            let key_bytes = bytes
                .get(cursor.position() as usize..start)
                .ok_or_else(|| invalid("metadata key truncated".to_string()))?;
            let key = std::str::from_utf8(key_bytes)
                .map_err(|_| invalid("metadata key is not valid UTF-8".to_string()))?
                .to_owned();
            cursor.set_position(start as u64);

            let tag = cursor
                .read_u8()
                .map_err(|_| invalid("metadata value tag truncated".to_string()))?;
            let value = match tag {
                TAG_BOOL => {
                    let byte = cursor
                        .read_u8()
                        .map_err(|_| invalid("metadata bool value truncated".to_string()))?;
                    match byte {
                        0 => MetaValue::Bool(false),
                        1 => MetaValue::Bool(true),
                        _ => return Err(invalid(format!("invalid metadata bool byte: {byte}"))),
                    }
                }
                TAG_INT => MetaValue::Int(
                    cursor
                        .read_i64::<LittleEndian>()
                        .map_err(|_| invalid("metadata int value truncated".to_string()))?,
                ),
                TAG_FLOAT => MetaValue::Float(
                    cursor
                        .read_f64::<LittleEndian>()
                        .map_err(|_| invalid("metadata float value truncated".to_string()))?,
                ),
                TAG_STR => {
                    let str_len = cursor
                        .read_u32::<LittleEndian>()
                        .ok()
                        .and_then(|len| usize::try_from(len).ok())
                        .ok_or_else(|| invalid("metadata string length invalid".to_string()))?;
                    let start = usize::try_from(cursor.position())
                        .ok()
                        .and_then(|pos| pos.checked_add(str_len))
                        .ok_or_else(|| invalid("metadata string length overflow".to_string()))?;
                    let str_bytes = bytes
                        .get(cursor.position() as usize..start)
                        .ok_or_else(|| invalid("metadata string value truncated".to_string()))?;
                    let value = std::str::from_utf8(str_bytes)
                        .map_err(|_| invalid("metadata string is not valid UTF-8".to_string()))?
                        .to_owned();
                    cursor.set_position(start as u64);
                    MetaValue::Str(value)
                }
                _ => return Err(invalid(format!("unknown metadata value tag: {tag}"))),
            };

            if map.insert(key, value).is_some() {
                return Err(invalid("duplicate metadata key".to_string()));
            }
        }

        if cursor.position() as usize != bytes.len() {
            return Err(invalid("trailing bytes after metadata blob".to_string()));
        }

        Ok(Self(map))
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
    fn test_binary_roundtrip_all_value_types() {
        let mut md = Metadata::new();
        md.insert("bool", true);
        md.insert("int", -42i64);
        md.insert("float", 2.5f64);
        md.insert("str", "hello wörld");
        md.insert("empty_str", "");

        let bytes = md.to_bytes();
        let parsed = Metadata::from_bytes(&bytes).unwrap();
        assert_eq!(md, parsed);
    }

    #[test]
    fn test_binary_roundtrip_empty_metadata() {
        let md = Metadata::new();
        let bytes = md.to_bytes();
        assert_eq!(bytes, vec![METADATA_ENCODING_VERSION, 0, 0, 0, 0]);
        assert_eq!(Metadata::from_bytes(&bytes).unwrap(), md);
    }

    #[test]
    fn test_binary_from_bytes_rejects_malformed_input() {
        let valid = sample_metadata().to_bytes();

        // Every truncated prefix must error, never panic.
        for len in 0..valid.len() {
            assert!(
                Metadata::from_bytes(&valid[..len]).is_err(),
                "prefix of length {len} unexpectedly parsed"
            );
        }

        // Empty input, unknown version, unknown tag, trailing garbage.
        assert!(Metadata::from_bytes(&[]).is_err());
        let mut bad_version = valid.clone();
        bad_version[0] = 99;
        assert!(Metadata::from_bytes(&bad_version).is_err());
        let mut bad_tag = valid.clone();
        // The last entry is an Int: tag byte sits 9 bytes before the end
        // (1 tag + 8 payload).
        let tag_pos = bad_tag.len() - 9;
        bad_tag[tag_pos] = 77;
        assert!(Metadata::from_bytes(&bad_tag).is_err());
        let mut trailing = valid.clone();
        trailing.push(0);
        assert!(Metadata::from_bytes(&trailing).is_err());
    }

    #[test]
    fn test_binary_from_bytes_rejects_invalid_bool_byte() {
        let md = {
            let mut md = Metadata::new();
            md.insert("b", true);
            md
        };
        let mut bytes = md.to_bytes();
        // Layout: version(1) count(4) key_len(4) "b"(1) tag(1) bool(1).
        *bytes.last_mut().unwrap() = 2;
        assert!(Metadata::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_binary_from_bytes_rejects_non_utf8_key() {
        let mut bytes = vec![METADATA_ENCODING_VERSION];
        bytes.extend_from_slice(&1_u32.to_le_bytes()); // one entry
        bytes.extend_from_slice(&1_u32.to_le_bytes()); // key length 1
        bytes.push(0xFF); // invalid UTF-8
        bytes.push(TAG_INT);
        bytes.extend_from_slice(&0_i64.to_le_bytes());
        assert!(Metadata::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_binary_from_bytes_rejects_duplicate_keys() {
        let mut bytes = vec![METADATA_ENCODING_VERSION];
        bytes.extend_from_slice(&2_u32.to_le_bytes()); // two entries
        for _ in 0..2 {
            bytes.extend_from_slice(&1_u32.to_le_bytes()); // key length 1
            bytes.extend_from_slice(b"k");
            bytes.push(TAG_INT);
            bytes.extend_from_slice(&0_i64.to_le_bytes());
        }
        assert!(Metadata::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_binary_from_bytes_absurd_length_does_not_allocate() {
        // key_len = u32::MAX with no key bytes present must fail fast.
        let mut bytes = vec![METADATA_ENCODING_VERSION];
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(Metadata::from_bytes(&bytes).is_err());
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
