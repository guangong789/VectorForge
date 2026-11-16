#include "vectorforge/scalar_reference.hpp"

#include "vectorforge/column_batch.hpp"
#include "vectorforge/embedding_buffer.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

using namespace vectorforge;

static_assert(!std::is_constructible_v<NumericConstant, double>);
static_assert(!std::is_constructible_v<NumericConstant, std::uint64_t>);
static_assert(std::same_as<ColumnBatch::ColumnView,
    std::variant<std::span<const std::int32_t>, std::span<const std::int64_t>, std::span<const float>>>);
template <typename T>
concept CanInspectColumn = requires(T&& columns) { std::forward<T>(columns).column_view(0); };
static_assert(CanInspectColumn<const ColumnBatch&> && !CanInspectColumn<ColumnBatch>);

void check(bool condition, std::string_view message,
           const std::source_location location = std::source_location::current()) {
    if (!condition) {
        throw std::runtime_error(std::string(location.file_name()) + ":" +
                                 std::to_string(location.line()) + ": " + std::string(message));
    }
}

template <typename Error, typename Function>
void expect_throw(Function&& function,
                  const std::source_location location = std::source_location::current()) {
    try {
        std::forward<Function>(function)();
    } catch (const Error&) {
        return;
    } catch (const std::exception& error) {
        check(false, std::string("unexpected exception: ") + error.what(), location);
    }
    check(false, "expected exception was not thrown", location);
}

void expect_results(const std::vector<ScoredRecord>& actual, const std::vector<ScoredRecord>& expected,
                    const std::source_location location = std::source_location::current()) {
    check(actual == expected, "result IDs, scores and order must match hand-calculated expectations", location);
}

struct Fixture {
    ColumnBatch columns{{std::vector<std::int32_t>{10, 20, 30, 20, 40},
                         std::vector<std::int64_t>{9'007'199'254'740'993LL, 9'007'199'254'740'992LL,
                             9'007'199'254'740'994LL, 9'007'199'254'740'995LL, 9'007'199'254'740'996LL},
                         std::vector<float>{0.5F, 1.5F, -1.0F, 2.0F, 3.0F}}};
    EmbeddingBuffer embeddings{5, 3, {1, 2, 0,  2, 0, 1,  -1, 1, 3,  0, 4, -1,  3, -2, 2}};
    std::array<RecordID, 5> ids{42, 7, 90, 11, 3};
    std::array<float, 3> query{2, -1, 1};
    // Dataset-row scores are 0, 5, 0, -5, 10. IDs intentionally differ from row indices.

    std::vector<ScoredRecord> run(std::span<const Predicate> predicates = {}, std::size_t k = 100) const {
        return execute_scalar_reference(columns, embeddings, ids, query, predicates, k);
    }
};

void column_inspection() {
    const Fixture f;
    check(std::get<std::span<const std::int32_t>>(f.columns.column_view(0)).data() ==
          f.columns.column<std::int32_t>(0).data(), "int32 inspection borrows original storage");
    check(std::get<std::span<const std::int64_t>>(f.columns.column_view(1))[0] == 9'007'199'254'740'993LL,
          "int64 inspection retains exact value");
    check(std::get<std::span<const float>>(f.columns.column_view(2)).data() == f.columns.column<float>(2).data(),
          "float inspection borrows original storage");
    expect_throw<std::out_of_range>([&] { (void)f.columns.column_view(3); });
    const ColumnBatch empty({});
    expect_throw<std::out_of_range>([&] { (void)empty.column_view(0); });
}

void basic_top_k() {
    const Fixture f;
    const std::array predicates{Predicate{0, Comparison::greater_equal, std::int32_t{20}}};
    expect_results(f.run(predicates, 2), {{3, 10}, {7, 5}});
    expect_results(f.run({}, 1), {{3, 10}});
    expect_results(f.run(), {{3, 10}, {7, 5}, {42, 0}, {90, 0}, {11, -5}});
    expect_results(f.run({}, std::numeric_limits<std::size_t>::max()),
                   {{3, 10}, {7, 5}, {42, 0}, {90, 0}, {11, -5}});
}

void and_predicates() {
    const Fixture f;
    const std::array pair{Predicate{0, Comparison::greater_equal, std::int32_t{20}},
                          Predicate{2, Comparison::less_equal, 2.0F}};
    expect_results(f.run(pair), {{7, 5}, {90, 0}, {11, -5}});
    const std::array triple{pair[0], pair[1],
        Predicate{1, Comparison::not_equal, std::int64_t{9'007'199'254'740'995LL}}};
    expect_results(f.run(triple), {{7, 5}, {90, 0}});
    const std::array impossible{Predicate{0, Comparison::greater, std::int32_t{40}}};
    expect_results(f.run(impossible), {});
    const std::array contradictory{Predicate{0, Comparison::less, std::int32_t{20}},
                                   Predicate{0, Comparison::greater, std::int32_t{30}}};
    expect_results(f.run(contradictory), {});
    expect_results(f.run({}, 0), {});
}

void all_comparisons_and_types() {
    const ColumnBatch columns({std::vector<std::int32_t>{-2, 0, 2},
        std::vector<std::int64_t>{9'007'199'254'740'992LL, 9'007'199'254'740'993LL, 9'007'199'254'740'994LL},
        std::vector<float>{-0.5F, 0.5F, 1.5F}});
    const EmbeddingBuffer embeddings(3, 1, {0, 0, 0});
    const std::array<RecordID, 3> ids{30, 10, 20};
    const std::array query{1.0F};
    const std::array<NumericConstant, 3> constants{std::int32_t{0}, std::int64_t{9'007'199'254'740'993LL}, 0.5F};
    const std::array cases{
        std::pair{Comparison::less, std::vector<ScoredRecord>{{30, 0}}},
        std::pair{Comparison::less_equal, std::vector<ScoredRecord>{{10, 0}, {30, 0}}},
        std::pair{Comparison::greater, std::vector<ScoredRecord>{{20, 0}}},
        std::pair{Comparison::greater_equal, std::vector<ScoredRecord>{{10, 0}, {20, 0}}},
        std::pair{Comparison::equal, std::vector<ScoredRecord>{{10, 0}}},
        std::pair{Comparison::not_equal, std::vector<ScoredRecord>{{20, 0}, {30, 0}}},
    };
    for (std::size_t column = 0; column < constants.size(); ++column) {
        for (const auto& [op, expected] : cases) {
            const std::array predicates{Predicate{column, op, constants[column]}};
            expect_results(execute_scalar_reference(columns, embeddings, ids, query, predicates, 10), expected);
        }
    }
}

void exact_integer_extremes() {
    const ColumnBatch columns({std::vector<std::int64_t>{std::numeric_limits<std::int64_t>::min(),
                                                       std::numeric_limits<std::int64_t>::max()}});
    const EmbeddingBuffer embeddings(2, 1, {1, 2});
    const std::array<RecordID, 2> ids{0, std::numeric_limits<RecordID>::max()};
    const std::array query{1.0F};
    const std::array predicates{Predicate{0, Comparison::equal, std::numeric_limits<std::int64_t>::max()}};
    expect_results(execute_scalar_reference(columns, embeddings, ids, query, predicates, 2),
                   {{std::numeric_limits<RecordID>::max(), 2}});
}

void deterministic_ties() {
    const ColumnBatch columns({std::vector<std::int32_t>{0, 0, 0}});
    const EmbeddingBuffer embeddings(3, 2, {1, 2, 2, 1, 3, 0});
    const std::array<RecordID, 3> ids{90, 10, 50};
    const std::array query{1.0F, 1.0F};
    expect_results(execute_scalar_reference(columns, embeddings, ids, query, {}, 2), {{10, 3}, {50, 3}});
    expect_results(execute_scalar_reference(columns, embeddings, ids, query, {}, 3), {{10, 3}, {50, 3}, {90, 3}});

    const EmbeddingBuffer zero_embeddings(3, 1, {-std::numeric_limits<float>::denorm_min(), 0.0F, -0.0F});
    const std::array half_query{0.5F};
    expect_results(execute_scalar_reference(columns, zero_embeddings, ids, half_query, {}, 3),
                   {{10, 0}, {50, 0}, {90, 0}});
    const EmbeddingBuffer negatives(3, 1, {-3, -1, -2});
    const std::array one_query{1.0F};
    expect_results(execute_scalar_reference(columns, negatives, ids, one_query, {}, 3),
                   {{10, -1}, {50, -2}, {90, -3}});
}

void duplicate_ids() {
    const Fixture f;
    auto ids = f.ids;
    ids[4] = ids[0];
    const std::array impossible{Predicate{0, Comparison::greater, std::int32_t{100}}};
    for (const auto k : std::array<std::size_t, 2>{0, 2}) {
        expect_throw<std::invalid_argument>([&] {
            (void)execute_scalar_reference(f.columns, f.embeddings, ids, f.query, {}, k);
        });
        expect_throw<std::invalid_argument>([&] {
            (void)execute_scalar_reference(f.columns, f.embeddings, ids, f.query, impossible, k);
        });
    }
}

void dimensions_and_alignment() {
    const Fixture f;
    const EmbeddingBuffer short_embeddings(1, 3, {1, 2, 3});
    const std::array short_query{1.0F, 2.0F};
    const std::array long_query{1.0F, 2.0F, 3.0F, 4.0F};
    for (const auto k : std::array<std::size_t, 2>{0, 2}) {
        expect_throw<std::invalid_argument>([&] {
            (void)execute_scalar_reference(f.columns, short_embeddings, f.ids, f.query, {}, k);
        });
        expect_throw<std::invalid_argument>([&] {
            (void)execute_scalar_reference(f.columns, f.embeddings, std::span(f.ids).first(4), f.query, {}, k);
        });
        expect_throw<std::invalid_argument>([&] {
            (void)execute_scalar_reference(f.columns, f.embeddings, f.ids, short_query, {}, k);
        });
        expect_throw<std::invalid_argument>([&] {
            (void)execute_scalar_reference(f.columns, f.embeddings, f.ids, long_query, {}, k);
        });
        expect_throw<std::invalid_argument>([&] {
            (void)execute_scalar_reference(f.columns, f.embeddings, f.ids, {}, {}, k);
        });
    }
}

void invalid_predicates() {
    const Fixture f;
    const std::array bad_types{Predicate{0, Comparison::equal, 20.0F},
        Predicate{0, Comparison::equal, std::int64_t{20}},
        Predicate{1, Comparison::equal, std::int32_t{20}},
        Predicate{2, Comparison::equal, std::int32_t{2}}};
    for (const auto k : std::array<std::size_t, 2>{0, 2}) {
        for (const auto& predicate : bad_types) {
            expect_throw<std::invalid_argument>([&] { (void)f.run(std::span(&predicate, 1), k); });
        }
        const std::array bad_column{Predicate{3, Comparison::equal, std::int32_t{0}}};
        expect_throw<std::out_of_range>([&] { (void)f.run(bad_column, k); });
        const std::array huge_column{Predicate{std::numeric_limits<std::size_t>::max(), Comparison::equal, 0.0F}};
        expect_throw<std::out_of_range>([&] { (void)f.run(huge_column, k); });
        const std::array bad_op{Predicate{0, static_cast<Comparison>(99), std::int32_t{20}}};
        expect_throw<std::invalid_argument>([&] { (void)f.run(bad_op, k); });
        // Validation must inspect a later predicate even if the first would reject every row.
        const std::array unreachable{Predicate{0, Comparison::greater, std::int32_t{100}}, bad_types[0]};
        expect_throw<std::invalid_argument>([&] { (void)f.run(unreachable, k); });
    }
}

void non_finite_inputs() {
    const Fixture f;
    const std::array bad_values{std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    const std::array no_matches{Predicate{0, Comparison::greater, std::int32_t{100}}};
    for (const float bad : bad_values) {
        auto query = f.query;
        query[2] = bad;
        const ColumnBatch bad_columns({std::vector<std::int32_t>{10, 20, 30, 20, 40},
                                        std::vector<float>{0, 0, 0, 0, bad}});
        const EmbeddingBuffer bad_embeddings(5, 3, {1, 2, 0, 2, 0, 1, -1, 1, 3, 0, 4, -1, 3, -2, bad});
        const std::array bad_constant{Predicate{2, Comparison::less, bad}};
        for (const auto k : std::array<std::size_t, 2>{0, 3}) {
            expect_throw<std::invalid_argument>([&] {
                (void)execute_scalar_reference(f.columns, f.embeddings, f.ids, query, no_matches, k);
            });
            // Float column is unreferenced by the filter, and its row would be filtered out.
            expect_throw<std::invalid_argument>([&] {
                (void)execute_scalar_reference(bad_columns, f.embeddings, f.ids, f.query, no_matches, k);
            });
            expect_throw<std::invalid_argument>([&] {
                (void)execute_scalar_reference(f.columns, bad_embeddings, f.ids, f.query, no_matches, k);
            });
            expect_throw<std::invalid_argument>([&] { (void)f.run(bad_constant, k); });
        }
    }
}

void non_finite_scores_and_zero_k_policy() {
    const ColumnBatch columns({std::vector<std::int32_t>{1}});
    const std::array<RecordID, 1> ids{8};
    constexpr float maximum = std::numeric_limits<float>::max();
    const EmbeddingBuffer product_overflow(1, 1, {maximum});
    const std::array twice{2.0F};
    expect_throw<std::overflow_error>([&] {
        (void)execute_scalar_reference(columns, product_overflow, ids, twice, {}, 1);
    });
    const EmbeddingBuffer negative_overflow(1, 1, {-maximum});
    expect_throw<std::overflow_error>([&] {
        (void)execute_scalar_reference(columns, negative_overflow, ids, twice, {}, 1);
    });
    const EmbeddingBuffer sum_overflow(1, 2, {maximum, maximum});
    const std::array ones{1.0F, 1.0F};
    expect_throw<std::overflow_error>([&] {
        (void)execute_scalar_reference(columns, sum_overflow, ids, ones, {}, 1);
    });
    // All inputs are finite: K=0 validates them but performs no scoring.
    expect_results(execute_scalar_reference(columns, sum_overflow, ids, ones, {}, 0), {});
    const std::array no_match{Predicate{0, Comparison::equal, std::int32_t{0}}};
    expect_results(execute_scalar_reference(columns, sum_overflow, ids, ones, no_match, 1), {});
    const EmbeddingBuffer finite_maximum(1, 1, {maximum});
    const std::array one{1.0F};
    expect_results(execute_scalar_reference(columns, finite_maximum, ids, one, {}, 1), {{8, maximum}});
}

void accumulation_policy() {
    const ColumnBatch columns({std::vector<std::int32_t>{0}});
    const std::array<RecordID, 1> ids{6};
    const EmbeddingBuffer cancellation(1, 3, {16'777'216.0F, 1.0F, -16'777'216.0F});
    const std::array ones{1.0F, 1.0F, 1.0F};
    // A float32 accumulator loses +1 at 2^24; double accumulation would yield 1.
    expect_results(execute_scalar_reference(columns, cancellation, ids, ones, {}, 1), {{6, 0}});
    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    const EmbeddingBuffer fused(1, 2, {-1.0F, 1.0F + epsilon});
    const std::array query{1.0F, 1.0F - epsilon};
    // (1+2^-23)(1-2^-23)-1 = -2^-46 with one fused rounding; separate operations yield 0.
    expect_results(execute_scalar_reference(columns, fused, ids, query, {}, 1), {{6, -0x1p-46F}});
}

void empty_and_single_row() {
    const ColumnBatch empty({});
    const EmbeddingBuffer no_embeddings(0, 2, {});
    const std::array query{1.0F, -1.0F};
    expect_results(execute_scalar_reference(empty, no_embeddings, {}, query, {}, 5), {});
    expect_results(execute_scalar_reference(empty, no_embeddings, {}, query, {}, 0), {});
    expect_throw<std::invalid_argument>([&] {
        (void)execute_scalar_reference(empty, no_embeddings, {}, {}, {}, 0);
    });
    const ColumnBatch typed_empty({std::vector<float>{}});
    const std::array predicate{Predicate{0, Comparison::equal, 1.0F}};
    expect_results(execute_scalar_reference(typed_empty, no_embeddings, {}, query, predicate, 5), {});
    const std::array wrong_type{Predicate{0, Comparison::equal, std::int32_t{1}}};
    expect_throw<std::invalid_argument>([&] {
        (void)execute_scalar_reference(typed_empty, no_embeddings, {}, query, wrong_type, 0);
    });
    expect_throw<std::out_of_range>([&] {
        (void)execute_scalar_reference(empty, no_embeddings, {}, query, predicate, 0);
    });
    const ColumnBatch one_column({std::vector<float>{1}});
    const EmbeddingBuffer one_embedding(1, 2, {4, 3});
    const std::array<RecordID, 1> id{123};
    expect_results(execute_scalar_reference(one_column, one_embedding, id, query, predicate, 10), {{123, 1}});
    expect_results(execute_scalar_reference(one_column, one_embedding, id, query, predicate, 0), {});
}

void owned_results_and_repeat_calls() {
    const auto result = [] {
        const Fixture local;
        return local.run({}, 2);
    }();
    expect_results(result, {{3, 10}, {7, 5}});
    const Fixture f;
    const std::array query{-2.0F, 1.0F, -1.0F};
    expect_results(f.run({}, 1), {{3, 10}});
    expect_results(execute_scalar_reference(f.columns, f.embeddings, f.ids, query, {}, 1), {{11, 5}});
    expect_results(f.run({}, 1), {{3, 10}});
    check(f.columns.column<std::int32_t>(0)[0] == 10 && f.embeddings.row(0)[1] == 2,
          "queries leave backing inputs unchanged");
}

} // namespace

int main() {
    const std::array tests{
        std::pair{"read-only column inspection", column_inspection},
        std::pair{"basic Top-K and row association", basic_top_k},
        std::pair{"AND predicates and no matches", and_predicates},
        std::pair{"all comparisons across all numeric types", all_comparisons_and_types},
        std::pair{"integer and Record ID extremes", exact_integer_extremes},
        std::pair{"deterministic ties, zero and negative scores", deterministic_ties},
        std::pair{"duplicate Record IDs", duplicate_ids},
        std::pair{"dimension and row alignment validation", dimensions_and_alignment},
        std::pair{"invalid predicates", invalid_predicates},
        std::pair{"non-finite inputs, including unused data", non_finite_inputs},
        std::pair{"score overflow and K=0 validation policy", non_finite_scores_and_zero_k_policy},
        std::pair{"documented float32 accumulation", accumulation_policy},
        std::pair{"empty and single-row datasets", empty_and_single_row},
        std::pair{"owned results and independent calls", owned_results_and_repeat_calls},
    };
    std::size_t failed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS: " << name << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "FAIL: " << name << ": " << error.what() << '\n';
        } catch (...) {
            ++failed;
            std::cerr << "FAIL: " << name << ": unknown exception\n";
        }
    }
    std::cout << (tests.size() - failed) << '/' << tests.size() << " test groups passed\n";
    return failed == 0 ? 0 : 1;
}
