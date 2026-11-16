#include "scoring/candidate_scoring.hpp"

#include "vectorforge/batched_executor.hpp"
#include "vectorforge/column_batch.hpp"
#include "vectorforge/embedding_buffer.hpp"
#include "vectorforge/scalar_reference.hpp"
#include "vectorforge/validated_dataset.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <source_location>
#include <stdexcept>
#include <string>
#include <utility>

// Fail the first one-record candidate-vector allocation, not exception-message
// storage. All fixtures/oracles are prepared before arming this test-only hook.
namespace allocation_failure {
bool armed = false;
std::size_t hits = 0;
}

void* operator new(std::size_t bytes) {
    if (allocation_failure::armed && bytes == sizeof(vectorforge::ScoredRecord)) {
        allocation_failure::armed = false;
        ++allocation_failure::hits;
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(bytes == 0 ? 1 : bytes)) { return p; }
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {
using namespace vectorforge;
constexpr auto W = detail::candidate_window_capacity;
constexpr auto maximum = std::numeric_limits<std::size_t>::max();
constexpr std::array<std::size_t, 7> batch_sizes{1, 7, W - 1, W, W + 1, 2 * W + 5, maximum};
std::size_t comparisons = 0;

void check(bool condition, const char* message,
           const std::source_location where = std::source_location::current()) {
    if (!condition) {
        throw std::runtime_error(std::string(where.file_name()) + ":" + std::to_string(where.line()) + ": " + message);
    }
}

template <typename Error, typename Function>
void expect_throw(Function&& function) {
    try { function(); }
    catch (const Error&) { return; }
    throw std::runtime_error("expected exception was not thrown");
}

void exact(std::span<const ScoredRecord> actual, std::span<const ScoredRecord> expected) {
    check(actual.size() == expected.size(), "result size");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        check(actual[i].id == expected[i].id &&
              std::bit_cast<std::uint32_t>(actual[i].score) == std::bit_cast<std::uint32_t>(expected[i].score),
              "global row/ID association, order and float32 bits");
    }
    ++comparisons;
}

void differential(const ValidatedDataset& dataset, std::span<const float> query,
                  std::span<const Predicate> predicates, std::size_t matches) {
    for (const auto k : std::array<std::size_t, 5>{0, 1, matches, matches + 3, maximum}) {
        const auto expected = execute_scalar_reference(dataset.columns(), dataset.embeddings(), dataset.record_ids(),
                                                        query, predicates, k);
        check(expected.size() == std::min(k, matches), "known matching cardinality");
        exact(execute_scalar_reference(dataset, query, predicates, k), expected);
        for (const auto batch : batch_sizes) {
            exact(execute_batched(dataset, query, predicates, k, batch), expected);
            exact(execute_batched(dataset.columns(), dataset.embeddings(), dataset.record_ids(), query, predicates, k, batch),
                  expected);
        }
    }
}

ValidatedDataset fixture(std::size_t matches, std::size_t dimension, bool sparse) {
    const auto rows = sparse ? matches * 3 + 2 : matches;
    std::vector<std::int32_t> flags(rows), tags(rows, 4);
    std::vector<float> values(rows * dimension);
    std::vector<RecordID> ids(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        flags[row] = !sparse || (row % 3 == 1 && row / 3 < matches) ? 1 : 0;
        ids[row] = static_cast<RecordID>((rows - row) * 17 + 7);
        for (std::size_t j = 0; j < dimension; ++j) {
            values[row * dimension + j] = static_cast<float>(static_cast<int>((row + j) % 5) - 2) / 8;
        }
    }
    const ColumnBatch columns({std::move(flags), std::move(tags)});
    const EmbeddingBuffer embeddings(rows, dimension, std::move(values));
    return ValidatedDataset(columns, embeddings, ids);
}

void windows_batches_and_mapping() {
    const std::array predicates{Predicate{0, Comparison::equal, std::int32_t{1}},
                                Predicate{1, Comparison::greater_equal, std::int32_t{4}}};
    for (const auto matches : std::array<std::size_t, 7>{0, 1, W - 1, W, W + 1, 2 * W + 1, 3 * W + 3}) {
        for (const bool sparse : {false, true}) {
            const auto dataset = fixture(matches, 5, sparse);
            const std::array query{1.0F, -1.0F, 0.5F, -0.5F, 1.0F};
            differential(dataset, query, predicates, matches);
            differential(dataset, query, {}, dataset.columns().row_count());
            const std::array none{Predicate{1, Comparison::less, std::int32_t{0}}};
            differential(dataset, query, none, 0);
        }
    }
    const auto ties = fixture(2 * W + 1, 3, false);
    const std::array zero_query{0.0F, -0.0F, 0.0F};
    differential(ties, zero_query, {}, ties.columns().row_count());
    const auto result = execute_batched(ties, zero_query, {}, 1, W + 1);
    check(result[0].id == ties.record_ids().back(), "tie winner ID is independent of arrival/window order");
}

void direct_scoring_and_prefix() {
    const EmbeddingBuffer embeddings(5, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, 2, 0, 1});
    const ColumnBatch columns({std::vector<std::int32_t>(5, 0)});
    const std::array<RecordID, 5> ids{0, 1, 2, 3, 4};
    const std::array query{1.0F, -1.0F, 0.5F};
    const auto records = execute_scalar_reference(columns, embeddings, ids, query, {}, 5);
    const std::array<std::size_t, 3> selected{4, 0, 2};
    std::array<float, 3> scores{};
    check(detail::score_candidates(embeddings, query, selected, scores) == 3, "all selected rows scored");
    for (std::size_t i = 0; i < selected.size(); ++i) {
        const auto expected = std::find_if(records.begin(), records.end(), [&](const auto& r) { return r.id == selected[i]; });
        check(expected != records.end() && std::bit_cast<std::uint32_t>(scores[i]) == std::bit_cast<std::uint32_t>(expected->score),
              "private boundary preserves caller row-slot order, including noncontiguous/nonmonotone rows");
    }
    check(detail::score_candidates(embeddings, query, {}, {}) == 0, "empty private window");
    expect_throw<std::invalid_argument>([&] { (void)detail::score_candidates(embeddings, query, selected, {}); });
    expect_throw<std::invalid_argument>([&] { (void)detail::score_candidates(embeddings, {}, {}, {}); });
    const std::array<std::size_t, 1> invalid_row{5};
    std::array<float, 1> one{};
    expect_throw<std::out_of_range>([&] { (void)detail::score_candidates(embeddings, query, invalid_row, one); });
    const std::array<std::size_t, W + 1> too_many{};
    std::array<float, W + 1> too_many_scores{};
    expect_throw<std::invalid_argument>([&] { (void)detail::score_candidates(embeddings, query, too_many, too_many_scores); });
    const EmbeddingBuffer overflow(3, 1, {1, std::numeric_limits<float>::max(), 3});
    const std::array twice{2.0F};
    const std::array<std::size_t, 3> order{0, 1, 2};
    scores.fill(-99);
    check(detail::score_candidates(overflow, twice, order, scores) == 1 && scores[0] == 2 && scores[1] == -99 && scores[2] == -99,
          "nonfinite intermediate yields exact finite prefix and does not score later rows");
}

void numerical_boundaries() {
    const float a = std::ldexp(1.0F, -23), denorm = std::numeric_limits<float>::denorm_min();
    const auto rows = 2 * W + 1;
    std::vector<std::int32_t> flags(rows, 1);
    std::vector<float> values(rows * 5, 0);
    std::vector<RecordID> ids(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        ids[row] = static_cast<RecordID>(9000 - row * 13);
        if (row % 3 == 0) { values[row * 5] = 1; values[row * 5 + 1] = 1 - a; }
        else if (row % 3 == 1) { values[row * 5 + 2] = -0.0F; }
        else { values[row * 5 + 2] = denorm; }
    }
    for (const auto row : std::array<std::size_t, 2>{W - 1, W + 1}) {
        flags[row] = 0;
        values[row * 5 + 2] = std::numeric_limits<float>::max();
    }
    const ColumnBatch columns({std::move(flags)});
    const EmbeddingBuffer embeddings(rows, 5, std::move(values));
    const ValidatedDataset dataset(columns, embeddings, ids);
    const std::array query{-1.0F, 1.0F + a, 2.0F, 0.0F, -0.0F};
    const std::array predicates{Predicate{0, Comparison::equal, std::int32_t{1}}};
    differential(dataset, query, predicates, rows - 2);
    const auto scored = execute_batched(dataset, query, predicates, maximum, W + 1);
    check(std::ranges::any_of(scored, [&](const auto& r) { return r.score == -std::ldexp(1.0F, -46); }),
          "cancellation residual survives window scoring");
    check(std::ranges::any_of(scored, [&](const auto& r) { return r.score == 2 * denorm; }), "gradual underflow preserved");

    for (const auto error_row : std::array<std::size_t, 5>{0, W - 1, W, W + 1, 2 * W}) {
        std::vector<float> huge(rows, 1);
        huge[error_row] = std::numeric_limits<float>::max();
        const ColumnBatch all({std::vector<std::int32_t>(rows, 1)});
        const EmbeddingBuffer data(rows, 1, std::move(huge));
        const ValidatedDataset admitted(all, data, ids);
        const std::array twice{2.0F};
        check(execute_scalar_reference(admitted, twice, {}, 0).empty(), "reference K=0 skips overflow scoring");
        expect_throw<std::overflow_error>([&] { (void)execute_scalar_reference(admitted, twice, {}, 1); });
        for (const auto batch : batch_sizes) {
            exact(execute_batched(admitted, twice, {}, 0, batch), {});
            exact(execute_batched(all, data, ids, twice, {}, 0, batch), {});
            expect_throw<std::overflow_error>([&] { (void)execute_batched(admitted, twice, {}, 1, batch); });
            expect_throw<std::overflow_error>([&] { (void)execute_batched(all, data, ids, twice, {}, 1, batch); });
        }
    }
}

void overflow_and_allocation_order() {
    for (const bool first_overflows : {false, true}) {
        const ColumnBatch columns({std::vector<std::int32_t>{1, 1, 1}});
        const float largest = std::numeric_limits<float>::max();
        const EmbeddingBuffer embeddings(3, 1, first_overflows ? std::vector<float>{largest, 1, 1}
                                                             : std::vector<float>{1, largest, 1});
        const std::array<RecordID, 3> ids{3, 2, 1};
        const ValidatedDataset dataset(columns, embeddings, ids);
        const std::array query{2.0F};
        for (const bool trusted : {false, true}) {
            // Arm only the trusted path, avoiding legacy ID-set allocations.
            // Both entry points check ordinary overflow below.
            if (trusted) {
                allocation_failure::hits = 0;
                allocation_failure::armed = true;
                bool expected = false;
                try { (void)execute_batched(dataset, query, {}, 1, 3); }
                catch (const std::bad_alloc&) { expected = !first_overflows; }
                catch (const std::overflow_error&) { expected = first_overflows; }
                catch (...) { allocation_failure::armed = false; throw; }
                allocation_failure::armed = false;
                check(expected && allocation_failure::hits == (first_overflows ? 0U : 1U),
                      "earlier candidate allocation failure retains precedence; first overflow retains precedence");
            }
            try {
                if (trusted) { (void)execute_batched(dataset, query, {}, 1, 3); }
                else { (void)execute_batched(columns, embeddings, ids, query, {}, 1, 3); }
                throw std::runtime_error("missing overflow");
            } catch (const std::overflow_error& error) {
                check(std::string(error.what()) == "batched dot product produced a non-finite score", "existing diagnostic preserved");
            }
        }
    }
}

void fixed_seed_differential() {
    std::mt19937_64 random(0x56463106ULL);
    for (std::size_t trial = 0; trial < 24; ++trial) {
        const auto rows = static_cast<std::size_t>(random() % (3 * W + 19));
        const auto dimension = static_cast<std::size_t>(random() % 35 + 1);
        std::vector<std::int32_t> flags(rows);
        std::vector<float> values(rows * dimension), query(dimension);
        std::vector<RecordID> ids(rows);
        std::size_t matches = 0;
        for (std::size_t row = 0; row < rows; ++row) {
            flags[row] = random() % 4 == 0 ? 1 : 0;
            matches += flags[row] == 1 ? 1U : 0U;
            ids[row] = static_cast<RecordID>(1'000'000 + (rows - row) * 31);
        }
        auto next = [&] { return static_cast<float>(static_cast<int>(random() % 33) - 16) / 16; };
        for (auto& v : values) { v = next(); }
        for (auto& v : query) { v = next(); }
        const ColumnBatch columns({std::move(flags)});
        const EmbeddingBuffer embeddings(rows, dimension, std::move(values));
        const ValidatedDataset dataset(columns, embeddings, ids);
        const std::array predicates{Predicate{0, Comparison::equal, std::int32_t{1}}};
        differential(dataset, query, predicates, matches);
    }
}

} // namespace

int main() {
    try {
        windows_batches_and_mapping();
        direct_scoring_and_prefix();
        numerical_boundaries();
        overflow_and_allocation_order();
        fixed_seed_differential();
        std::cout << "PASS bounded scoring: W=" << W << ", " << comparisons
                  << " exact differential comparisons, tails/mapping/finite-prefix/exception precedence\n";
        return 0;
    } catch (const std::exception& error) {
        allocation_failure::armed = false;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
