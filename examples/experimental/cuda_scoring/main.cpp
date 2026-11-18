#include "scoring.hpp"

#include <vectorforge/batched_executor.hpp>
#include <vectorforge/column_batch.hpp>
#include <vectorforge/embedding_buffer.hpp>
#include <vectorforge/scalar_reference.hpp>
#include <vectorforge/validated_dataset.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace {
using namespace cuda_scoring_pilot;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }

Score host_score(std::span<const float> row, std::span<const float> query) {
    float accumulator = +0.0f;
    for (std::size_t d = 0; d < row.size(); ++d) {
        accumulator = std::fma(query[d], row[d], accumulator);
        if (!std::isfinite(accumulator)) { return {0.0f, static_cast<std::uint32_t>(d)}; }
    }
    return {accumulator, no_failure};
}

void dump_bits(const char* name, std::span<const float> values) {
    std::fprintf(stderr, "%s", name);
    for (const float value : values) { std::fprintf(stderr, " %08" PRIx32, bits(value)); }
    std::fprintf(stderr, "\n");
}

std::vector<Score> verify(const char* label, ResidentScorer& scorer,
                          std::span<const float> embeddings, std::span<const RowIndex> rows,
                          std::span<const float> query) {
    auto actual = scorer.score(rows, query);
    require(actual.size() == rows.size(), "result slot count changed");
    std::printf("fixture=%s rows=%zu dimension=%zu candidates=%zu\n",
                label, embeddings.size() / query.size(), query.size(), rows.size());
    for (std::size_t slot = 0; slot < rows.size(); ++slot) {
        const auto row = embeddings.subspan(static_cast<std::size_t>(rows[slot]) * query.size(), query.size());
        const auto expected = host_score(row, query);
        const auto result = actual[slot];
        std::printf("comparison fixture=%s slot=%zu row=%" PRIu32
                    " cpu_bits=%08" PRIx32 " gpu_bits=%08" PRIx32
                    " cpu_failure=%" PRIu32 " gpu_failure=%" PRIu32 "\n",
                    label, slot, rows[slot], bits(expected.value), bits(result.value),
                    expected.first_failure, result.first_failure);
        if (bits(expected.value) != bits(result.value) || expected.first_failure != result.first_failure) {
            std::fprintf(stderr, "numerical_mismatch fixture=%s slot=%zu row=%" PRIu32 "\n", label, slot, rows[slot]);
            dump_bits("row_bits", row);
            dump_bits("query_bits", query);
            throw std::runtime_error("Contract G bitwise/status mismatch");
        }
    }
    std::printf("fixture=%s status=verified\n", label);
    return actual;
}

template<class Error, class Action>
void reject(const char* label, Action&& action) {
    try { action(); }
    catch (const Error& error) {
        std::printf("adapter_rejection fixture=%s diagnostic=%s\n", label, error.what());
        return;
    }
    throw std::runtime_error("expected adapter rejection");
}

void smoke() {
    const std::array row{1.0f, -2.0f, 3.0f};
    const std::array query{0.5f, 1.0f, -1.0f};
    const std::array<RowIndex, 1> selected{0};
    ResidentScorer scorer(row, 1, row.size(), selected.size());
    const auto actual = verify("lifecycle", scorer, row, selected, query);
    require(bits(actual[0].value) == bits(-4.5f), "hand-computed smoke score");
    scorer.close();
    scorer.close(); // Successful cleanup is idempotent.
    reject<std::logic_error>("use_after_close", [&] { (void)scorer.score(selected, query); });
}

void ordinary_and_rejections() {
    constexpr std::size_t row_count = 137, dimension = 7;
    std::array<float, row_count * dimension> ordered{};
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        ordered[i] = static_cast<float>(static_cast<int>((i * 17) % 37) - 18) / 8.0f;
    }
    std::array<RowIndex, 133> selected{};
    for (std::size_t i = 0; i < selected.size(); ++i) {
        selected[i] = static_cast<RowIndex>((i * 11) % row_count);
    }
    selected[1] = selected.back() = static_cast<RowIndex>(row_count - 1);
    const std::array query{1.0f, -0.5f, 0.25f, -2.0f, 0.0f, -0.0f, 0.75f};
    const auto scorer = [&] {
        std::vector<float> source(ordered.begin(), ordered.end());
        auto resident = std::make_unique<ResidentScorer>(source, row_count, dimension, selected.size());
        std::fill(source.begin(), source.end(), 0.0f);
        return resident; // Resident storage outlives the original copy.
    }();
    (void)verify("zero_candidates", *scorer, ordered, {}, query);
    reject<std::invalid_argument>("empty_query", [&] { (void)scorer->score({}, {}); });
    reject<std::invalid_argument>("dimension_mismatch", [&] {
        (void)scorer->score(selected, std::span(query).first(dimension - 1));
    });
    auto nonfinite_query = query;
    nonfinite_query.back() = std::numeric_limits<float>::quiet_NaN();
    reject<std::invalid_argument>("nonfinite_query", [&] { (void)scorer->score(selected, nonfinite_query); });
    const std::array<RowIndex, 2> outside{0, static_cast<RowIndex>(row_count)};
    reject<std::out_of_range>("boundary_rejection", [&] { (void)scorer->score(outside, query); });
    const std::array<RowIndex, selected.size() + 1> over_capacity{};
    reject<std::invalid_argument>("candidate_capacity", [&] { (void)scorer->score(over_capacity, query); });
    (void)verify("ordered_duplicates_non_aligned", *scorer, ordered, selected, query);
    auto second_query = query;
    std::reverse(second_query.begin(), second_query.end());
    (void)verify("resident_second_query", *scorer, ordered, selected, second_query);
    scorer->close();
}

void special_scores() {
    const float tiny = std::numeric_limits<float>::denorm_min();
    const std::array denormals{-tiny, -0.0f, +0.0f, tiny, -2 * tiny, 2 * tiny};
    {
        ResidentScorer scorer(denormals, denormals.size(), 1, denormals.size());
        const std::array<RowIndex, 6> rows{0, 1, 2, 3, 4, 5};
        const std::array query{0.5f};
        const auto actual = verify("signed_zero_gradual_underflow", scorer, denormals, rows, query);
        const std::array<std::uint32_t, 6> expected{0x80000000U, 0, 0, 0, 0x80000001U, 1};
        for (std::size_t i = 0; i < expected.size(); ++i) {
            require(bits(actual[i].value) == expected[i], "hand-computed signed-zero/subnormal bits");
        }
        scorer.close();
    }
    const float maximum = std::numeric_limits<float>::max();
    const std::array values{maximum, 0.0f, -maximum, maximum / 2, maximum, -maximum,
                            maximum / 4, maximum / 4, maximum, maximum / 2, 0.0f, 0.0f,
                            -maximum, 0.0f, 0.0f};
    ResidentScorer scorer(values, 5, 3, 6);
    const std::array<RowIndex, 6> rows{3, 2, 0, 1, 3, 4};
    const std::array query{2.0f, 1.0f, 1.0f};
    const auto actual = verify("overflow_at_different_dimensions", scorer, values, rows, query);
    const std::array<std::uint32_t, 6> expected{no_failure, 2, 0, 1, no_failure, 0};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(actual[i].first_failure == expected[i], "hand-computed failure dimension");
    }
    require(bits(actual.front().value) == bits(maximum), "finite value near overflow");
    const auto first_failed = std::find_if(actual.begin(), actual.end(),
        [](const Score& result) { return result.first_failure != no_failure; });
    require(first_failed == actual.begin() + 1 && first_failed->first_failure == 2,
            "find earliest failing slot by input order");
    std::printf("numerical_failure first_slot=1 row=2 first_dimension=2 result_published=false\n");
    scorer.close();
}

void cpu_overlap() {
    namespace vf = vectorforge;
    const float epsilon = std::numeric_limits<float>::epsilon();
    const std::array values{-1.0f, 1.0f + epsilon, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                            1.0f + epsilon, 0.0f, 0.0f, -3.0f, 0.0f, 0.0f};
    const std::array query{1.0f, 1.0f - epsilon, 1.0f};
    const auto snapshot = [&] {
        const vf::ColumnBatch columns({std::vector<std::int32_t>{1, 1, 1, 1, 0}});
        const vf::EmbeddingBuffer embeddings(5, 3, std::vector<float>(values.begin(), values.end()));
        const std::array<vf::RecordID, 5> ids{50, 40, 7, 90, 3};
        return vf::ValidatedDataset(columns, embeddings, ids);
    }();
    std::vector<float> upload;
    for (std::size_t row = 0; row < snapshot.embeddings().row_count(); ++row) {
        const auto values_view = snapshot.embeddings().row(row);
        upload.insert(upload.end(), values_view.begin(), values_view.end());
    }
    ResidentScorer scorer(upload, 5, 3, 4);
    const std::array<RowIndex, 4> rows{0, 1, 2, 3};
    const auto actual = verify("explicit_fma_and_ties", scorer, values, rows, query);
    require(bits(actual[0].value) == bits(-0x1p-46f), "fused cancellation residual");
    require(bits(actual[1].value) == bits(actual[2].value) &&
                bits(actual[3].value) == bits(1.0f + epsilon), "equal and adjacent float32 scores");
    std::array<vf::ScoredRecord, 4> recovered{};
    for (std::size_t i = 0; i < rows.size(); ++i) {
        require(actual[i].first_failure == no_failure, "unexpected failing score in CPU overlap");
        recovered[i] = {snapshot.record_ids()[rows[i]], actual[i].value};
    }
    std::sort(recovered.begin(), recovered.end(), [](const auto& a, const auto& b) {
        return a.score > b.score || (a.score == b.score && a.id < b.id);
    });
    const std::array predicates{vf::Predicate{0, vf::Comparison::equal, std::int32_t{1}}};
    const auto bounded = vf::execute_batched(snapshot, query, predicates, rows.size(), 3);
    const auto full_sort = vf::execute_scalar_reference(snapshot, query, predicates, rows.size());
    require(bounded.size() == rows.size() && full_sort.size() == rows.size(), "CPU overlap cardinality");
    const std::array<vf::RecordID, 4> ids{90, 7, 40, 50};
    for (std::size_t i = 0; i < rows.size(); ++i) {
        require(recovered[i].id == ids[i] && bounded[i].id == ids[i] && full_sort[i].id == ids[i] &&
                    bits(recovered[i].score) == bits(bounded[i].score) &&
                    bits(recovered[i].score) == bits(full_sort[i].score), "CPU overlap match by ID and bits");
        std::printf("cpu_overlap rank=%zu id=%" PRIu64 " score_bits=%08" PRIx32 "\n",
                    i, ids[i], bits(recovered[i].score));
    }
    scorer.close();
}

void adapter_and_cleanup() {
    const std::array row{1.0f};
    const std::array<RowIndex, 1> selected{0};
    reject<std::invalid_argument>("zero_dimension", [&] { ResidentScorer bad({}, 0, 0, 0); });
    reject<std::invalid_argument>("embedding_shape", [&] { ResidentScorer bad(row, 2, 1, 1); });
    const std::array inf{std::numeric_limits<float>::infinity()};
    reject<std::invalid_argument>("nonfinite_embedding", [&] { ResidentScorer bad(inf, 1, 1, 1); });
    reject<std::length_error>("checked_byte_overflow", [] {
        ResidentScorer bad({}, no_failure, no_failure, 0);
    });
    reject<std::length_error>("device_budget", [&] { ResidentScorer bad(row, 1, 1, 16 * 1024 * 1024); });
    reject<std::length_error>("pinned_budget", [&] { ResidentScorer bad(row, 1, 1, 512 * 1024); });
    {
        ResidentScorer empty({}, 0, row.size(), 0);
        (void)verify("empty_dataset", empty, {}, {}, row);
        empty.close();
    }
    struct Unwind {};
    try {
        ResidentScorer scoped(row, 1, 1, 1);
        (void)verify("before_scope_unwind", scoped, row, selected, row);
        throw Unwind{};
    } catch (const Unwind&) {
        std::printf("cleanup fixture=host_unwind status=released\n");
    }
    ResidentScorer next(row, 1, 1, 1);
    (void)verify("after_scope_unwind", next, row, selected, row);
    next.close();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto mode = argc == 2 ? std::string_view(argv[1]) : std::string_view{};
        require(mode == "--smoke" || mode == "--test", "usage: vectorforge_cuda_scoring_pilot --smoke|--test");
        require(std::fesetround(FE_TONEAREST) == 0, "host round-to-nearest unavailable");
#if defined(__x86_64__)
        require((_mm_getcsr() & 0x8040U) == 0, "host FTZ/DAZ must be off");
#endif
        select_device();
        if (mode == "--smoke") {
            smoke();
        } else {
            ordinary_and_rejections();
            special_scores();
            cpu_overlap();
            adapter_and_cleanup();
        }
        std::printf("PASS %s\n", argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "pilot_failed: %s\n", error.what());
        return 1;
    }
}
