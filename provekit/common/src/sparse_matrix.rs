use {
    crate::{FieldElement, InternedFieldElement, Interner},
    ark_std::Zero,
    rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    serde::{
        de::{SeqAccess, Visitor},
        ser::SerializeStruct,
        Deserialize, Deserializer, Serialize, Serializer,
    },
    std::{
        fmt::{self, Debug},
        ops::{Mul, Range},
    },
};

#[derive(Debug, Clone, Copy)]
pub struct DeltaEncodingStats {
    pub total_entries:  usize,
    pub absolute_bytes: usize,
    pub delta_bytes:    usize,
}

impl DeltaEncodingStats {
    pub const fn savings_bytes(&self) -> usize {
        self.absolute_bytes.saturating_sub(self.delta_bytes)
    }

    pub fn savings_percent(&self) -> f64 {
        if self.absolute_bytes == 0 {
            0.0
        } else {
            self.savings_bytes() as f64 / self.absolute_bytes as f64 * 100.0
        }
    }
}

const fn varint_size(value: u32) -> usize {
    match value {
        0..=0x7f => 1,
        0x80..=0x3fff => 2,
        0x4000..=0x1f_ffff => 3,
        0x20_0000..=0xfff_ffff => 4,
        _ => 5,
    }
}

/// A sparse matrix with interned field elements.
///
/// Uses delta encoding for column indices during serialization to reduce size.
/// Within each row, the first column index is stored as absolute, and
/// subsequent columns are stored as deltas from the previous column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseMatrix {
    /// The number of rows in the matrix.
    pub num_rows: usize,

    /// The number of columns in the matrix.
    pub num_cols: usize,

    // List of indices in `col_indices` such that the column index is the start of a new row.
    new_row_indices: Vec<u32>,

    /// List of column indices that have values (absolute values in memory).
    col_indices: Vec<u32>,

    /// List of values.
    values: Vec<InternedFieldElement>,
}

/// Convert absolute column indices to delta-encoded indices per row.
fn encode_col_deltas(
    col_indices: &[u32],
    new_row_indices: &[u32],
    total_entries: usize,
) -> Vec<u32> {
    let mut deltas = Vec::with_capacity(col_indices.len());
    let num_rows = new_row_indices.len();

    for row in 0..num_rows {
        let start = new_row_indices[row] as usize;
        let end = new_row_indices
            .get(row + 1)
            .map_or(total_entries, |&v| v as usize);

        let row_cols = &col_indices[start..end];
        if row_cols.is_empty() {
            continue;
        }

        debug_assert!(
            row_cols.windows(2).all(|w| w[0] <= w[1]),
            "Column indices must be sorted within each row"
        );

        // First column is stored as absolute
        deltas.push(row_cols[0]);

        // Subsequent columns stored as delta from previous
        for i in 1..row_cols.len() {
            deltas.push(row_cols[i] - row_cols[i - 1]);
        }
    }

    deltas
}

/// Convert delta-encoded column indices back to absolute indices per row.
fn decode_col_deltas(deltas: &[u32], new_row_indices: &[u32], total_entries: usize) -> Vec<u32> {
    let mut col_indices = Vec::with_capacity(deltas.len());
    let num_rows = new_row_indices.len();

    let mut delta_idx = 0;
    for row in 0..num_rows {
        let start = new_row_indices[row] as usize;
        let end = new_row_indices
            .get(row + 1)
            .map_or(total_entries, |&v| v as usize);

        let row_len = end - start;
        if row_len == 0 {
            continue;
        }

        // First column is absolute
        let first_col = deltas[delta_idx];
        col_indices.push(first_col);
        delta_idx += 1;

        // Subsequent columns are cumulative deltas
        let mut prev_col = first_col;
        for _ in 1..row_len {
            let col = prev_col + deltas[delta_idx];
            col_indices.push(col);
            prev_col = col;
            delta_idx += 1;
        }
    }

    col_indices
}

impl Serialize for SparseMatrix {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let col_deltas =
            encode_col_deltas(&self.col_indices, &self.new_row_indices, self.values.len());

        let mut state = serializer.serialize_struct("SparseMatrix", 5)?;
        state.serialize_field("num_rows", &self.num_rows)?;
        state.serialize_field("num_cols", &self.num_cols)?;
        state.serialize_field("new_row_indices", &self.new_row_indices)?;
        state.serialize_field("col_deltas", &col_deltas)?;
        state.serialize_field("values", &self.values)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for SparseMatrix {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            NumRows,
            NumCols,
            NewRowIndices,
            ColDeltas,
            Values,
        }

        struct SparseMatrixVisitor;

        impl<'de> Visitor<'de> for SparseMatrixVisitor {
            type Value = SparseMatrix;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct SparseMatrix")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<SparseMatrix, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let num_rows = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let num_cols = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let new_row_indices: Vec<u32> = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                let col_deltas: Vec<u32> = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;
                let values: Vec<InternedFieldElement> = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(4, &self))?;

                let col_indices = decode_col_deltas(&col_deltas, &new_row_indices, values.len());

                Ok(SparseMatrix {
                    num_rows,
                    num_cols,
                    new_row_indices,
                    col_indices,
                    values,
                })
            }

            fn visit_map<V>(self, mut map: V) -> Result<SparseMatrix, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut num_rows = None;
                let mut num_cols = None;
                let mut new_row_indices: Option<Vec<u32>> = None;
                let mut col_deltas: Option<Vec<u32>> = None;
                let mut values: Option<Vec<InternedFieldElement>> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::NumRows => {
                            if num_rows.is_some() {
                                return Err(serde::de::Error::duplicate_field("num_rows"));
                            }
                            num_rows = Some(map.next_value()?);
                        }
                        Field::NumCols => {
                            if num_cols.is_some() {
                                return Err(serde::de::Error::duplicate_field("num_cols"));
                            }
                            num_cols = Some(map.next_value()?);
                        }
                        Field::NewRowIndices => {
                            if new_row_indices.is_some() {
                                return Err(serde::de::Error::duplicate_field("new_row_indices"));
                            }
                            new_row_indices = Some(map.next_value()?);
                        }
                        Field::ColDeltas => {
                            if col_deltas.is_some() {
                                return Err(serde::de::Error::duplicate_field("col_deltas"));
                            }
                            col_deltas = Some(map.next_value()?);
                        }
                        Field::Values => {
                            if values.is_some() {
                                return Err(serde::de::Error::duplicate_field("values"));
                            }
                            values = Some(map.next_value()?);
                        }
                    }
                }

                let num_rows =
                    num_rows.ok_or_else(|| serde::de::Error::missing_field("num_rows"))?;
                let num_cols =
                    num_cols.ok_or_else(|| serde::de::Error::missing_field("num_cols"))?;
                let new_row_indices = new_row_indices
                    .ok_or_else(|| serde::de::Error::missing_field("new_row_indices"))?;
                let col_deltas =
                    col_deltas.ok_or_else(|| serde::de::Error::missing_field("col_deltas"))?;
                let values = values.ok_or_else(|| serde::de::Error::missing_field("values"))?;

                let col_indices = decode_col_deltas(&col_deltas, &new_row_indices, values.len());

                Ok(SparseMatrix {
                    num_rows,
                    num_cols,
                    new_row_indices,
                    col_indices,
                    values,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "num_rows",
            "num_cols",
            "new_row_indices",
            "col_deltas",
            "values",
        ];
        deserializer.deserialize_struct("SparseMatrix", FIELDS, SparseMatrixVisitor)
    }
}

/// A hydrated sparse matrix with uninterned field elements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HydratedSparseMatrix<'a> {
    pub matrix: &'a SparseMatrix,
    interner:   &'a Interner,
}

impl SparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            num_rows:        rows,
            num_cols:        cols,
            new_row_indices: vec![0; rows],
            col_indices:     Vec::new(),
            values:          Vec::new(),
        }
    }

    pub const fn hydrate<'a>(&'a self, interner: &'a Interner) -> HydratedSparseMatrix<'a> {
        HydratedSparseMatrix {
            matrix: self,
            interner,
        }
    }

    pub fn num_entries(&self) -> usize {
        self.values.len()
    }

    pub fn delta_encoding_stats(&self) -> DeltaEncodingStats {
        let deltas = encode_col_deltas(&self.col_indices, &self.new_row_indices, self.values.len());

        let absolute_bytes: usize = self.col_indices.iter().map(|&v| varint_size(v)).sum();
        let delta_bytes: usize = deltas.iter().map(|&v| varint_size(v)).sum();

        DeltaEncodingStats {
            total_entries: self.col_indices.len(),
            absolute_bytes,
            delta_bytes,
        }
    }

    pub fn grow(&mut self, rows: usize, cols: usize) {
        // TODO: Make it default infinite size instead.
        assert!(rows >= self.num_rows);
        assert!(cols >= self.num_cols);
        self.num_rows = rows;
        self.num_cols = cols;
        self.new_row_indices.resize(rows, self.values.len() as u32);
    }

    /// Set the value at the given row and column.
    pub fn set(&mut self, row: usize, col: usize, value: InternedFieldElement) {
        assert!(row < self.num_rows, "row index out of bounds");
        assert!(col < self.num_cols, "column index out of bounds");

        // Find the row
        let row_range = self.row_range(row);
        let cols = &self.col_indices[row_range.clone()];

        // Find the column
        match cols.binary_search(&(col as u32)) {
            Ok(_) => {
                unreachable!("Duplicate column {col} in row {row}");
            }
            Err(i) => {
                // Need to insert column at i
                let i = i + row_range.start;
                self.col_indices.insert(i, col as u32);
                self.values.insert(i, value);
                for index in &mut self.new_row_indices[row + 1..] {
                    *index += 1;
                }
            }
        }
    }

    /// Iterate over the non-default entries of a row of the matrix.
    pub fn iter_row(
        &self,
        row: usize,
    ) -> impl Iterator<Item = (usize, InternedFieldElement)> + use<'_> {
        let row_range = self.row_range(row);
        let cols = self.col_indices[row_range.clone()].iter().copied();
        let values = self.values[row_range].iter().copied();
        cols.zip(values).map(|(col, value)| (col as usize, value))
    }

    /// Iterate over the non-default entries of the matrix.
    pub fn iter(&self) -> impl Iterator<Item = ((usize, usize), InternedFieldElement)> + use<'_> {
        (0..self.new_row_indices.len()).flat_map(|row| {
            self.iter_row(row)
                .map(move |(col, value)| ((row, col), value))
        })
    }

    fn row_range(&self, row: usize) -> Range<usize> {
        let start = *self
            .new_row_indices
            .get(row)
            .expect("Row index out of bounds") as usize;
        let end = self
            .new_row_indices
            .get(row + 1)
            .map_or(self.values.len(), |&v| v as usize);
        start..end
    }

    /// Transpose the matrix, swapping rows and columns.
    ///
    /// Returns a new `SparseMatrix` where entry (i, j) in the original
    /// becomes (j, i) in the result. The interned values are preserved
    /// and remain valid for the same `Interner`.
    ///
    /// Uses a counting sort on the column index, which runs in time linear in
    /// the number of entries. Rows are visited in ascending order and each
    /// entry is appended to its column's bucket, so every row of the result
    /// lists its entries by ascending original row: the same (row, column)
    /// order, and therefore the same matrix, as sorting the entries.
    pub fn transpose(&self) -> SparseMatrix {
        // Prefix sums of per-column counts give each result row's start.
        let mut row_starts = vec![0u32; self.num_cols + 1];
        for row in 0..self.num_rows {
            for i in self.row_range(row) {
                row_starts[self.col_indices[i] as usize + 1] += 1;
            }
        }
        for col in 0..self.num_cols {
            row_starts[col + 1] += row_starts[col];
        }
        let nnz = row_starts[self.num_cols] as usize;
        row_starts.truncate(self.num_cols);
        let new_row_indices = row_starts;

        let mut next_slot = new_row_indices.clone();
        let mut col_indices = vec![0u32; nnz];
        let mut values = self
            .values
            .first()
            .map_or_else(Vec::new, |&value| vec![value; nnz]);
        for row in 0..self.num_rows {
            for i in self.row_range(row) {
                let slot = &mut next_slot[self.col_indices[i] as usize];
                let position = *slot as usize;
                debug_assert!(
                    position == new_row_indices[self.col_indices[i] as usize] as usize
                        || col_indices[position - 1] != row as u32,
                    "Duplicate (row, col) entries in sparse matrix transpose"
                );
                col_indices[position] = row as u32;
                values[position] = self.values[i];
                *slot += 1;
            }
        }

        SparseMatrix {
            num_rows: self.num_cols,
            num_cols: self.num_rows,
            new_row_indices,
            col_indices,
            values,
        }
    }

    /// Remap column indices using provided mapping function - in-place and
    /// parallel
    pub fn remap_columns<F>(&mut self, remap_fn: F)
    where
        F: Fn(usize) -> usize + Send + Sync,
    {
        // Step 1: Remap all column indices in parallel
        self.col_indices.par_iter_mut().for_each(|col| {
            *col = remap_fn(*col as usize) as u32;
        });

        // Step 2: Re-sort each row sequentially (fast enough, avoids unsafe)
        for row in 0..self.num_rows {
            let start = self.new_row_indices[row] as usize;
            let end = self
                .new_row_indices
                .get(row + 1)
                .map_or(self.col_indices.len(), |&v| v as usize);

            let row_cols = &mut self.col_indices[start..end];
            let row_vals = &mut self.values[start..end];

            let mut pairs: Vec<_> = row_cols
                .iter()
                .zip(row_vals.iter())
                .map(|(&c, &v)| (c, v))
                .collect();
            pairs.sort_unstable_by_key(|(c, _)| *c);

            for (i, (c, v)) in pairs.into_iter().enumerate() {
                row_cols[i] = c;
                row_vals[i] = v;
            }
        }
    }
}

impl HydratedSparseMatrix<'_> {
    /// Iterate over the non-default entries of a row of the matrix.
    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = (usize, FieldElement)> + use<'_> {
        self.matrix.iter_row(row).map(|(col, value)| {
            (
                col,
                self.interner.get(value).expect("Value not in interner."),
            )
        })
    }

    /// Iterate over the non-default entries of the matrix.
    pub fn iter(&self) -> impl Iterator<Item = ((usize, usize), FieldElement)> + use<'_> {
        self.matrix.iter().map(|((i, j), v)| {
            (
                (i, j),
                self.interner.get(v).expect("Value not in interner."),
            )
        })
    }
}

/// Right multiplication by vector (parallel over rows).
impl Mul<&[FieldElement]> for HydratedSparseMatrix<'_> {
    type Output = Vec<FieldElement>;

    fn mul(self, rhs: &[FieldElement]) -> Self::Output {
        assert_eq!(
            self.matrix.num_cols,
            rhs.len(),
            "Vector length does not match number of columns."
        );
        (0..self.matrix.num_rows)
            .into_par_iter()
            .map(|row| {
                self.iter_row(row)
                    .map(|(col, value)| value * rhs[col])
                    .fold(FieldElement::zero(), |acc, x| acc + x)
            })
            .collect()
    }
}

/// Left multiplication by vector (sequential scatter pattern).
///
/// The primary call site (`calculate_external_row_of_r1cs_matrices`)
/// now uses transpose + parallel right-multiply instead.
impl Mul<HydratedSparseMatrix<'_>> for &[FieldElement] {
    type Output = Vec<FieldElement>;

    fn mul(self, rhs: HydratedSparseMatrix<'_>) -> Self::Output {
        assert_eq!(
            self.len(),
            rhs.matrix.num_rows,
            "Vector length does not match number of rows."
        );
        let mut result = vec![FieldElement::zero(); rhs.matrix.num_cols];
        for ((i, j), value) in rhs.iter() {
            result[j] += value * self[i];
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_encoding_roundtrip() {
        let col_indices = vec![3, 15, 100, 5, 50, 200];
        let new_row_indices = vec![0, 3];
        let total_entries = 6;

        let deltas = encode_col_deltas(&col_indices, &new_row_indices, total_entries);
        let decoded = decode_col_deltas(&deltas, &new_row_indices, total_entries);

        assert_eq!(col_indices, decoded);
    }

    #[test]
    fn test_delta_encoding_values() {
        let col_indices = vec![3, 15, 100];
        let new_row_indices = vec![0];
        let total_entries = 3;

        let deltas = encode_col_deltas(&col_indices, &new_row_indices, total_entries);

        assert_eq!(deltas, vec![3, 12, 85]);
    }

    #[test]
    fn test_delta_encoding_multiple_rows() {
        let col_indices = vec![0, 10, 20, 5, 15];
        let new_row_indices = vec![0, 3];
        let total_entries = 5;

        let deltas = encode_col_deltas(&col_indices, &new_row_indices, total_entries);
        assert_eq!(deltas, vec![0, 10, 10, 5, 10]);

        let decoded = decode_col_deltas(&deltas, &new_row_indices, total_entries);
        assert_eq!(col_indices, decoded);
    }

    #[test]
    fn test_delta_encoding_empty_row() {
        let col_indices = vec![5, 10];
        let new_row_indices = vec![0, 0, 2];
        let total_entries = 2;

        let deltas = encode_col_deltas(&col_indices, &new_row_indices, total_entries);
        let decoded = decode_col_deltas(&deltas, &new_row_indices, total_entries);

        assert_eq!(col_indices, decoded);
    }

    /// Single non-zero in the whole matrix: one row, one column.
    #[test]
    fn test_delta_encoding_single_entry() {
        let col_indices = vec![42];
        let new_row_indices = vec![0];
        let total_entries = 1;

        let deltas = encode_col_deltas(&col_indices, &new_row_indices, total_entries);
        assert_eq!(deltas, vec![42]);

        let decoded = decode_col_deltas(&deltas, &new_row_indices, total_entries);
        assert_eq!(col_indices, decoded);
    }

    /// Each row has exactly one column; all deltas are absolute (no within-row
    /// deltas).
    #[test]
    fn test_delta_encoding_single_column_per_row() {
        let col_indices = vec![0, 5, 100];
        let new_row_indices = vec![0, 1, 2];
        let total_entries = 3;

        let deltas = encode_col_deltas(&col_indices, &new_row_indices, total_entries);
        assert_eq!(deltas, vec![0, 5, 100]);

        let decoded = decode_col_deltas(&deltas, &new_row_indices, total_entries);
        assert_eq!(col_indices, decoded);
    }

    /// Consecutive column indices (deltas of 1).
    #[test]
    fn test_delta_encoding_consecutive_columns() {
        let col_indices = vec![10, 11, 12, 13];
        let new_row_indices = vec![0];
        let total_entries = 4;

        let deltas = encode_col_deltas(&col_indices, &new_row_indices, total_entries);
        assert_eq!(deltas, vec![10, 1, 1, 1]);

        let decoded = decode_col_deltas(&deltas, &new_row_indices, total_entries);
        assert_eq!(col_indices, decoded);
    }

    /// All rows empty: no column indices, only row boundaries.
    #[test]
    fn test_delta_encoding_all_rows_empty() {
        let col_indices: Vec<u32> = vec![];
        let new_row_indices = vec![0, 0, 0];
        let total_entries = 0;

        let deltas = encode_col_deltas(&col_indices, &new_row_indices, total_entries);
        assert!(deltas.is_empty());

        let decoded = decode_col_deltas(&deltas, &new_row_indices, total_entries);
        assert!(decoded.is_empty());
    }

    /// Last row empty; only earlier rows have entries.
    #[test]
    fn test_delta_encoding_last_row_empty() {
        let col_indices = vec![1, 2, 7];
        let new_row_indices = vec![0, 2, 3];
        let total_entries = 3;

        let deltas = encode_col_deltas(&col_indices, &new_row_indices, total_entries);
        let decoded = decode_col_deltas(&deltas, &new_row_indices, total_entries);

        assert_eq!(col_indices, decoded);
    }

    /// Only one row has entries (row 2); rows 0, 1, 3 are empty. Roundtrip
    /// still works.
    #[test]
    fn test_delta_encoding_only_last_row_non_empty() {
        let col_indices = vec![3, 8];
        let new_row_indices = vec![0, 0, 0, 2];
        let total_entries = 2;

        let deltas = encode_col_deltas(&col_indices, &new_row_indices, total_entries);
        assert_eq!(deltas, vec![3, 5]);

        let decoded = decode_col_deltas(&deltas, &new_row_indices, total_entries);
        assert_eq!(col_indices, decoded);
    }

    /// Large column indices; deltas stay small (no u32 overflow in encoding).
    #[test]
    fn test_delta_encoding_large_column_indices() {
        let col_indices = vec![1_000_000, 1_000_001, 2_000_000];
        let new_row_indices = vec![0];
        let total_entries = 3;

        let deltas = encode_col_deltas(&col_indices, &new_row_indices, total_entries);
        assert_eq!(deltas, vec![1_000_000, 1, 999_999]);

        let decoded = decode_col_deltas(&deltas, &new_row_indices, total_entries);
        assert_eq!(col_indices, decoded);
    }

    #[test]
    fn test_sparse_matrix_serde_roundtrip() {
        let mut interner = Interner::new();
        let val1 = interner.intern(FieldElement::from(1u64));
        let val2 = interner.intern(FieldElement::from(2u64));
        let val3 = interner.intern(FieldElement::from(3u64));

        let mut matrix = SparseMatrix::new(3, 100);
        matrix.grow(3, 100);
        matrix.set(0, 5, val1);
        matrix.set(0, 20, val2);
        matrix.set(1, 50, val3);

        let serialized = postcard::to_allocvec(&matrix).expect("serialization failed");
        let deserialized: SparseMatrix =
            postcard::from_bytes(&serialized).expect("deserialization failed");

        assert_eq!(matrix, deserialized);
    }

    #[test]
    fn test_delta_encoding_size_reduction() {
        let mut interner = Interner::new();
        let val = interner.intern(FieldElement::from(1u64));

        let mut matrix = SparseMatrix::new(10, 1000);
        matrix.grow(10, 1000);

        for row in 0..10 {
            for col_offset in 0..20 {
                matrix.set(row, row * 50 + col_offset, val);
            }
        }

        let serialized = postcard::to_allocvec(&matrix).expect("serialization failed");

        let col_count = matrix.col_indices.len();
        let naive_col_bytes = col_count * 4;
        let actual_bytes = serialized.len();

        assert!(
            actual_bytes < naive_col_bytes,
            "delta encoding should reduce size: actual {} vs naive col bytes {}",
            actual_bytes,
            naive_col_bytes
        );
    }

    /// The comparison-sort transpose that `transpose` replaced, kept as the
    /// reference the counting sort must reproduce exactly.
    fn transpose_by_sort(matrix: &SparseMatrix) -> SparseMatrix {
        let mut entries: Vec<(u32, u32, InternedFieldElement)> = Vec::new();
        for row in 0..matrix.num_rows {
            for i in matrix.row_range(row) {
                entries.push((matrix.col_indices[i], row as u32, matrix.values[i]));
            }
        }
        entries.sort_unstable_by_key(|&(new_row, new_col, _)| (new_row, new_col));

        let mut new_row_indices = Vec::with_capacity(matrix.num_cols);
        let mut col_indices = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());
        let mut entry_idx = 0;
        for row in 0..matrix.num_cols {
            new_row_indices.push(entry_idx as u32);
            while entry_idx < entries.len() && entries[entry_idx].0 == row as u32 {
                col_indices.push(entries[entry_idx].1);
                values.push(entries[entry_idx].2);
                entry_idx += 1;
            }
        }
        SparseMatrix {
            num_rows: matrix.num_cols,
            num_cols: matrix.num_rows,
            new_row_indices,
            col_indices,
            values,
        }
    }

    /// SplitMix64, so the random matrices are reproducible without a test-only
    /// RNG dependency.
    fn next_random(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A `rows` x `cols` matrix in which each entry is present with probability
    /// `percent` / 100, holding one of `values` chosen at random.
    fn random_matrix(
        state: &mut u64,
        rows: usize,
        cols: usize,
        percent: u64,
        values: &[InternedFieldElement],
    ) -> SparseMatrix {
        let mut matrix = SparseMatrix::new(rows, cols);
        for row in 0..rows {
            for col in 0..cols {
                if next_random(state) % 100 < percent {
                    let value = values[next_random(state) as usize % values.len()];
                    matrix.set(row, col, value);
                }
            }
        }
        matrix
    }

    #[test]
    fn transpose_matches_sort_based_transpose() {
        let mut interner = Interner::new();
        let values: Vec<InternedFieldElement> = (1..=37u64)
            .map(|value| interner.intern(FieldElement::from(value)))
            .collect();
        let mut state = 0x7a5e_u64;
        let shapes = [
            (0, 0),
            (0, 5),
            (5, 0),
            (1, 1),
            (1, 50),
            (50, 1),
            (17, 33),
            (64, 64),
            (150, 200),
            (200, 150),
        ];
        for (rows, cols) in shapes {
            for percent in [0, 3, 30, 100] {
                let matrix = random_matrix(&mut state, rows, cols, percent, &values);
                let transposed = matrix.transpose();
                assert_eq!(
                    transposed,
                    transpose_by_sort(&matrix),
                    "counting-sort transpose differs from the sort-based transpose for a \
                     {rows}x{cols} matrix at {percent}% density"
                );
                assert_eq!(
                    transposed.transpose(),
                    matrix,
                    "transposing twice must return the original {rows}x{cols} matrix at \
                     {percent}% density"
                );
            }
        }
    }

    /// Rows added by `grow` after entries exist share the final row index,
    /// which the counting sort must handle like any other empty row.
    #[test]
    fn transpose_matches_sort_based_transpose_after_grow() {
        let mut interner = Interner::new();
        let values = [interner.intern(FieldElement::from(5u64))];
        let mut state = 0x9e0_u64;
        let mut matrix = random_matrix(&mut state, 20, 30, 25, &values);
        matrix.grow(45, 60);
        assert_eq!(
            matrix.transpose(),
            transpose_by_sort(&matrix),
            "counting-sort transpose differs from the sort-based transpose after grow"
        );
    }
}
