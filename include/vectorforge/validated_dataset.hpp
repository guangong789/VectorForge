#pragma once

#include "vectorforge/query_types.hpp"

#include <memory>
#include <span>

namespace vectorforge {

class ColumnBatch;
class EmbeddingBuffer;

// An owning, sealed snapshot admitted once for repeated trusted queries.
// Validates aligned row counts, unique IDs, and all stored float values' finiteness.
// Existing owners can contain aliases retained before vector moves, so admission
// deliberately copies payloads once. Moving those owners alone cannot revoke such
// aliases. Inputs need only stay alive and unchanged during this constructor call;
// subsequent source changes/destruction cannot affect the admitted snapshot.
//
// No mutable storage/assignment API is exposed. Moving this dataset transfers the
// sealed storage without copying or changing its address. Existing views remain
// valid while the destination lives; moved-from access throws logic_error. Views
// borrow storage and must not outlive its final owner. Rvalue getters are disabled.
//
// Let S = total stored bytes (column metadata and numeric/embedding/ID payloads),
// N = rows, M = matching rows.
// Admission: O(S) copying/scanning plus expected O(N) uniqueness work; O(N)
// temporary ID-set memory, freed before admission returns. Persistent storage:
// O(S), independent of original owners (which may then be released).
// Trusted query validation: O(D + predicates) time, O(1) auxiliary memory; no
// repeated dataset copy/ID set. Batched query candidate/output storage is
// O(min(K,M)) plus a fixed scoring window; scalar-reference candidates remain
// O(M), plus sort stack O(log M).
// Legacy raw-input queries still use their O(N) temporary duplicate-ID set.
// Existing E2E benchmarks continue to measure those legacy entry points.
class ValidatedDataset {
public:
    ValidatedDataset(const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
                     std::span<const RecordID> record_ids);
    ~ValidatedDataset();
    ValidatedDataset(const ValidatedDataset&) = delete;
    ValidatedDataset& operator=(const ValidatedDataset&) = delete;
    ValidatedDataset(ValidatedDataset&&) noexcept;
    ValidatedDataset& operator=(ValidatedDataset&&) = delete;

    [[nodiscard]] const ColumnBatch& columns() const &;
    const ColumnBatch& columns() const && = delete;
    [[nodiscard]] const EmbeddingBuffer& embeddings() const &;
    const EmbeddingBuffer& embeddings() const && = delete;
    [[nodiscard]] std::span<const RecordID> record_ids() const &;
    std::span<const RecordID> record_ids() const && = delete;

private:
    struct Storage;
    [[nodiscard]] const Storage& storage() const;
    std::unique_ptr<const Storage> storage_;
};

} // namespace vectorforge
