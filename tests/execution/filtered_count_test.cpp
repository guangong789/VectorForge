#include "vectorforge/filtered_count.hpp"

#include "vectorforge/batched_executor.hpp"
#include "vectorforge/column_batch.hpp"
#include "vectorforge/embedding_buffer.hpp"
#include "vectorforge/scalar_reference.hpp"
#include "vectorforge/validated_dataset.hpp"

#include <array>
#include <bit>
#include <iostream>
#include <limits>
#include <random>
#include <source_location>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using namespace vectorforge;
constexpr auto maximum = std::numeric_limits<std::size_t>::max();
constexpr std::int64_t big = 9'007'199'254'740'993LL;
constexpr std::array<std::size_t, 6> batch_sizes{1, 2, 4, 7, 999, maximum};
std::size_t count_comparisons = 0, topk_comparisons = 0;

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

// Independent oracle: ternary ordering on typed values, no production helper,
// batching, embedding access or similarity/Top-K calls to derive the count.
std::size_t oracle(const ColumnBatch& columns, std::span<const Predicate> predicates) {
    std::size_t result = 0;
    for (std::size_t row = 0; row < columns.row_count(); ++row) {
        bool all = true;
        for (const auto& predicate : predicates) {
            const bool matches = std::visit([&]<typename T>(T constant) {
                const T value = columns.column<T>(predicate.column_index)[row];
                const int order = static_cast<int>(value > constant) - static_cast<int>(value < constant);
                switch (predicate.comparison) {
                case Comparison::less: return order == -1;
                case Comparison::less_equal: return order != 1;
                case Comparison::greater: return order == 1;
                case Comparison::greater_equal: return order != -1;
                case Comparison::equal: return order == 0;
                case Comparison::not_equal: return order != 0;
                }
                throw std::runtime_error("oracle only accepts valid predicates");
            }, predicate.constant);
            all = all && matches;
        }
        if (all) { ++result; }
    }
    return result;
}

void count_batches(const ValidatedDataset& dataset, std::span<const Predicate> predicates, std::size_t expected) {
    check(oracle(dataset.columns(), predicates) == expected, "hand-calculated count agrees with independent oracle");
    const auto* ids = dataset.record_ids().data();
    for (const auto size : batch_sizes) {
        check(count_where(dataset, predicates, size) == expected, "COUNT across batch sizes");
        ++count_comparisons;
    }
    check(dataset.record_ids().data() == ids, "COUNT retains the same admitted storage");
}

void exact(std::span<const ScoredRecord> actual, std::span<const ScoredRecord> expected) {
    check(actual.size() == expected.size(), "Top-K count unchanged");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        check(actual[i].id == expected[i].id &&
              std::bit_cast<std::uint32_t>(actual[i].score) == std::bit_cast<std::uint32_t>(expected[i].score),
              "Top-K IDs/order/score bits unchanged after predicate extraction");
    }
    ++topk_comparisons;
}

void topk_paths(const ValidatedDataset& dataset, std::span<const float> query, std::span<const Predicate> predicates) {
    for (const auto k : std::array<std::size_t, 4>{0, 1, 7, maximum}) {
        const auto expected = execute_scalar_reference(dataset.columns(), dataset.embeddings(), dataset.record_ids(),
                                                        query, predicates, k);
        exact(execute_scalar_reference(dataset, query, predicates, k), expected);
        for (const auto size : std::array<std::size_t, 3>{1, 7, maximum}) {
            exact(execute_batched(dataset, query, predicates, k, size), expected);
            exact(execute_batched(dataset.columns(), dataset.embeddings(), dataset.record_ids(), query, predicates, k, size),
                  expected);
        }
    }
}

struct Fixture {
    ColumnBatch columns{{std::vector<std::int32_t>{-2, 0, 2, 2, 4, 6},
        std::vector<std::int64_t>{big, big + 1, big + 2, big + 3, big + 4, big + 5},
        std::vector<float>{-1, -0.0F, 0.0F, 0.5F, 1, 2}}};
    EmbeddingBuffer embeddings{6, 2, {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}};
    std::vector<RecordID> ids{90, 10, 70, 20, 60, 30};
    ValidatedDataset admit() const { return ValidatedDataset(columns, embeddings, ids); }
};

void known_counts_and_topk() {
    const Fixture f;
    const auto dataset = f.admit();
    count_batches(dataset, {}, 6);
    constexpr std::array comparisons{Comparison::less, Comparison::less_equal, Comparison::greater,
        Comparison::greater_equal, Comparison::equal, Comparison::not_equal};
    const std::array<std::array<std::size_t, 6>, 3> expected{{{2, 4, 2, 4, 2, 4}, {2, 3, 3, 4, 1, 5}, {1, 3, 3, 5, 2, 4}}};
    const std::array<NumericConstant, 3> constants{std::int32_t{2}, big + 2, 0.0F};
    for (std::size_t column = 0; column < 3; ++column) {
        for (std::size_t op = 0; op < comparisons.size(); ++op) {
            const std::array predicates{Predicate{column, comparisons[op], constants[column]}};
            count_batches(dataset, predicates, expected[column][op]);
            const std::array query{1.0F, 1.0F};
            topk_paths(dataset, query, predicates);
        }
    }
    const std::array partial{Predicate{0, Comparison::greater_equal, std::int32_t{2}},
                             Predicate{2, Comparison::less_equal, 1.0F}};
    count_batches(dataset, partial, 3);
    const std::array three{partial[0], partial[1], Predicate{1, Comparison::not_equal, big + 3}};
    count_batches(dataset, three, 2);
    const std::array none{Predicate{0, Comparison::less, std::int32_t{-10}}};
    const std::array all{Predicate{0, Comparison::greater_equal, std::int32_t{-2}}};
    count_batches(dataset, none, 0);
    count_batches(dataset, all, 6);
    for (const auto query : std::array{std::array{1.0F, 1.0F}, std::array{-1.0F, 0.0F}, std::array{0.0F, 0.0F}}) {
        topk_paths(dataset, query, {});
        topk_paths(dataset, query, partial);
        topk_paths(dataset, query, three);
        topk_paths(dataset, query, none);
    }
    const std::array query{1.0F, 1.0F};
    exact(execute_batched(dataset, query, {}, 3, 2), std::array<ScoredRecord, 3>{{{10, 1}, {20, 1}, {30, 1}}});
}

template <typename Error>
void reject_predicates(const ValidatedDataset& dataset, std::span<const Predicate> predicates) {
    expect_throw<Error>([&] { (void)count_where(dataset, predicates, 2); });
    const std::vector<float> query(dataset.embeddings().dimension(), 1.0F);
    for (const auto k : std::array<std::size_t, 2>{0, 1}) {
        expect_throw<Error>([&] { (void)execute_scalar_reference(dataset, query, predicates, k); });
        expect_throw<Error>([&] { (void)execute_scalar_reference(dataset.columns(), dataset.embeddings(),
            dataset.record_ids(), query, predicates, k); });
        expect_throw<Error>([&] { (void)execute_batched(dataset, query, predicates, k, 2); });
        expect_throw<Error>([&] { (void)execute_batched(dataset.columns(), dataset.embeddings(),
            dataset.record_ids(), query, predicates, k, 2); });
    }
}

void boundaries_and_rejection() {
    const Fixture f;
    const auto populated = f.admit();
    const ColumnBatch columns{{std::vector<std::int32_t>{}, std::vector<std::int64_t>{}, std::vector<float>{}}};
    const EmbeddingBuffer embeddings(0, 2, {});
    const ValidatedDataset empty(columns, embeddings, {});
    count_batches(empty, {}, 0);
    const std::array valid{Predicate{2, Comparison::greater_equal, -0.0F}};
    count_batches(empty, valid, 0);
    const ColumnBatch no_columns({});
    const ValidatedDataset no_columns_dataset(no_columns, embeddings, {});
    count_batches(no_columns_dataset, {}, 0);
    const std::array absent{Predicate{0, Comparison::equal, std::int32_t{0}}};
    reject_predicates<std::out_of_range>(no_columns_dataset, absent);

    for (const auto* dataset : {&populated, &empty}) {
        expect_throw<std::invalid_argument>([&] { (void)count_where(*dataset, {}, 0); });
        const std::array invalid_index{Predicate{3, Comparison::equal, std::int32_t{0}}};
        reject_predicates<std::out_of_range>(*dataset, invalid_index);
        for (const auto predicate : std::array{
            Predicate{0, Comparison::equal, std::int64_t{2}},
            Predicate{1, Comparison::equal, 2.0F},
            Predicate{2, Comparison::equal, std::int32_t{0}},
            Predicate{0, static_cast<Comparison>(255), std::int32_t{0}}}) {
            const std::array bad{predicate};
            reject_predicates<std::invalid_argument>(*dataset, bad);
        }
        for (const auto value : std::array{std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
            const std::array bad{Predicate{2, Comparison::equal, value}};
            reject_predicates<std::invalid_argument>(*dataset, bad);
        }
        const std::array none_then_bad{Predicate{0, Comparison::less, std::int32_t{-999}}, invalid_index[0]};
        reject_predicates<std::out_of_range>(*dataset, none_then_bad);
    }
    const ColumnBatch single{{std::vector<std::int32_t>{7}}};
    const EmbeddingBuffer huge(1, 1, {std::numeric_limits<float>::max()});
    const std::array<RecordID, 1> ids{std::numeric_limits<RecordID>::max()};
    const ValidatedDataset one(single, huge, ids);
    count_batches(one, {}, 1);
    const std::array matches{Predicate{0, Comparison::equal, std::int32_t{7}}};
    const std::array misses{Predicate{0, Comparison::not_equal, std::int32_t{7}}};
    count_batches(one, matches, 1);
    count_batches(one, misses, 0);
    const std::array query{2.0F};
    expect_throw<std::overflow_error>([&] { (void)execute_batched(one, query, matches, 1, 1); });
    check(count_where(one, matches, 1) == 1, "COUNT never scores even when a similarity query would overflow");
    auto moved_from = f.admit();
    const auto moved_to = std::move(moved_from);
    count_batches(moved_to, {}, 6);
    expect_throw<std::logic_error>([&] { (void)count_where(moved_from, {}, 1); });
}

void fixed_seed_differential() {
    std::mt19937_64 random(0x56463104ULL);
    for (std::size_t trial = 0; trial < 20; ++trial) {
        const auto rows = static_cast<std::size_t>(random() % 258);
        const auto dimension = static_cast<std::size_t>(random() % 5 + 1);
        const auto base = trial % 2 == 0 ? big : -big;
        std::vector<std::int32_t> ints(rows);
        std::vector<std::int64_t> longs(rows);
        std::vector<float> floats(rows), values(rows * dimension), query(dimension);
        std::vector<RecordID> ids(rows);
        auto small = [&] { return static_cast<int>(random() % 17) - 8; };
        for (std::size_t row = 0; row < rows; ++row) {
            ints[row] = static_cast<std::int32_t>(small());
            longs[row] = base + small();
            floats[row] = static_cast<float>(small()) / 4;
            ids[row] = static_cast<RecordID>(100'000 + rows - row);
        }
        for (auto& value : values) { value = static_cast<float>(small()) / 8; }
        for (auto& value : query) { value = static_cast<float>(small()) / 8; }
        const ColumnBatch columns({std::move(ints), std::move(longs), std::move(floats)});
        const EmbeddingBuffer embeddings(rows, dimension, std::move(values));
        const ValidatedDataset dataset(columns, embeddings, ids);
        for (std::size_t case_index = 0; case_index < 16; ++case_index) {
            std::vector<Predicate> predicates;
            const auto length = static_cast<std::size_t>(random() % 4);
            for (std::size_t i = 0; i < length; ++i) {
                const auto column = static_cast<std::size_t>(random() % 3);
                NumericConstant constant;
                if (column == 0) { constant = static_cast<std::int32_t>(small()); }
                else if (column == 1) { constant = base + small(); }
                else { constant = static_cast<float>(small()) / 4; }
                predicates.push_back({column, static_cast<Comparison>(random() % 6), constant});
            }
            const auto expected = oracle(columns, predicates);
            count_batches(dataset, predicates, expected);
            if (case_index % 4 == 0) { topk_paths(dataset, query, predicates); }
        }
    }
}

} // namespace

int main() {
    try {
        known_counts_and_topk();
        boundaries_and_rejection();
        fixed_seed_differential();
        std::cout << "PASS filtered COUNT: " << count_comparisons << " count checks; " << topk_comparisons
                  << " exact Top-K path comparisons; types/operators/AND/boundaries/rejections/independent oracle\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
