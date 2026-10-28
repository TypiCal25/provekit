use {
    crate::{utils::serde_ark, FieldElement},
    serde::{Deserialize, Serialize},
    std::collections::HashMap,
};

#[derive(Debug, Clone, Eq, Serialize, Deserialize)]
pub struct Interner {
    #[serde(with = "serde_ark")]
    values: Vec<FieldElement>,
    /// Reverse index: `FieldElement` → its position in `values`. Kept in sync
    /// by `intern`. Not serialized (the on-disk format remains a single
    /// `values` field for backward compatibility); rebuilt lazily on first
    /// `intern` after deserialize.
    #[serde(skip)]
    index:  HashMap<FieldElement, usize>,
}

/// `index` is `#[serde(skip)]` and rebuilt lazily, so a roundtripped
/// `Interner` has an empty map until the next `intern` call. Compare only
/// the canonical `values` field — handles based on `values` indices are
/// stable across map state, and the map is just an O(1) lookup cache.
impl PartialEq for Interner {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InternedFieldElement(usize);

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

impl Interner {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            index:  HashMap::new(),
        }
    }

    /// Intern a `FieldElement`, returning a stable handle.
    ///
    /// O(1) amortized — uses a `HashMap` to deduplicate. The map is populated
    /// lazily on first call after deserialize.
    pub fn intern(&mut self, value: FieldElement) -> InternedFieldElement {
        // After deserialize the map may be empty while `values` is not.
        // Rebuild once.
        if self.index.len() < self.values.len() {
            self.rebuild_index();
        }

        if let Some(&index) = self.index.get(&value) {
            return InternedFieldElement(index);
        }

        let index = self.values.len();
        self.values.push(value);
        self.index.insert(value, index);
        InternedFieldElement(index)
    }

    pub fn get(&self, el: InternedFieldElement) -> Option<FieldElement> {
        self.values.get(el.0).copied()
    }

    fn rebuild_index(&mut self) {
        self.index.clear();
        self.index.reserve(self.values.len());
        for (index, value) in self.values.iter().enumerate() {
            self.index.insert(*value, index);
        }
    }
}

#[cfg(test)]
mod tests {
    use {super::*, ark_ff::Field};

    /// Interning must hand out the same handles as the linear search it
    /// replaced: first occurrence wins, and repeats map to that index.
    #[test]
    fn intern_matches_first_occurrence_order() {
        let mut interner = Interner::new();
        let values: Vec<FieldElement> = (0..500u64)
            .map(|i| FieldElement::from(i % 97).pow([i % 5 + 1]))
            .collect();
        for value in &values {
            let handle = interner.intern(*value);
            let first = values.iter().position(|v| v == value).unwrap();
            let expected = values[..first]
                .iter()
                .fold(Vec::<FieldElement>::new(), |mut seen, v| {
                    if !seen.contains(v) {
                        seen.push(*v);
                    }
                    seen
                })
                .len();
            assert_eq!(
                handle,
                InternedFieldElement(expected),
                "a value's handle must be its first-occurrence position"
            );
            assert_eq!(interner.get(handle), Some(*value));
        }
    }

    /// After a serde roundtrip the index is empty; interning must rebuild it
    /// and keep existing handles stable.
    #[test]
    fn intern_after_roundtrip_reuses_existing_handles() {
        let mut interner = Interner::new();
        let handles: Vec<_> = (1..=20u64)
            .map(|i| interner.intern(FieldElement::from(i)))
            .collect();
        let bytes = postcard::to_stdvec(&interner).unwrap();
        let mut restored: Interner = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(restored, interner, "a roundtrip must preserve the values");
        for (i, handle) in (1..=20u64).zip(handles) {
            assert_eq!(
                restored.intern(FieldElement::from(i)),
                handle,
                "an existing value must keep its handle after a roundtrip"
            );
        }
        assert_eq!(
            restored.intern(FieldElement::from(21u64)),
            InternedFieldElement(20),
            "a new value must be appended after the existing ones"
        );
    }
}
