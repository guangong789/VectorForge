#include "vectorforge/batched_executor.hpp"

#include "vectorforge/column_batch.hpp"
#include "vectorforge/embedding_buffer.hpp"
#include "vectorforge/scalar_reference.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace vectorforge;
constexpr auto maximum_size = std::numeric_limits<std::size_t>::max();
constexpr std::array<std::size_t, 8> batch_sizes{1, 2, 3, 4, 5, 8, 9, maximum_size};
std::size_t differential_checks = 0;

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

// Compare float32 bit patterns, including signed zero. No epsilon or rounding of
// results is allowed to hide differences between the two sequential FMA paths.
void exact_results(const std::vector<ScoredRecord>& actual, const std::vector<ScoredRecord>& expected,
                   const std::source_location location = std::source_location::current()) {
    check(actual.size() == expected.size(), "candidate count differs", location);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        check(actual[i].id == expected[i].id, "Record ID or ordering differs", location);
        check(std::bit_cast<std::uint32_t>(actual[i].score) == std::bit_cast<std::uint32_t>(expected[i].score),
              "float32 score bits differ", location);
    }
}

void compare_paths(const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
                   std::span<const RecordID> ids, std::span<const float> query,
                   std::span<const Predicate> predicates, std::size_t k,
                   std::span<const std::size_t> sizes = batch_sizes) {
    const auto reference = execute_scalar_reference(columns, embeddings, ids, query, predicates, k);
    for (const auto size : sizes) {
        exact_results(execute_batched(columns, embeddings, ids, query, predicates, k, size), reference);
        ++differential_checks;
    }
}

template <typename Error>
void reject_in_both(const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
                    std::span<const RecordID> ids, std::span<const float> query,
                    std::span<const Predicate> predicates, std::size_t k) {
    expect_throw<Error>([&] { (void)execute_scalar_reference(columns, embeddings, ids, query, predicates, k); });
    for (const auto size : std::array<std::size_t, 3>{1, 3, maximum_size}) {
        expect_throw<Error>([&] { (void)execute_batched(columns, embeddings, ids, query, predicates, k, size); });
    }
}

struct Fixture {
    ColumnBatch columns{{std::vector<std::int32_t>{0, 1, 2, 3, 4, 5, 6, 7},
        std::vector<std::int64_t>{9'007'199'254'740'992LL, 9'007'199'254'740'993LL, 9'007'199'254'740'994LL,
            9'007'199'254'740'995LL, 9'007'199'254'740'996LL, 9'007'199'254'740'997LL,
            9'007'199'254'740'998LL, 9'007'199'254'740'999LL},
        std::vector<float>{0, 1, -1, 2, 0, 3, -2, 1}}};
    EmbeddingBuffer embeddings{8, 2, {1, 0, 0, 1, 2, 2, -1, 1, 3, 0, 0, -1, 1, 4, 4, 0}};
    std::array<RecordID, 8> ids{80, 40, 70, 10, 60, 30, 50, 20};
    std::array<float, 2> query{2, -1};
    // Row scores: 2, -1, 2, -3, 6, 1, -2, 8. The winner is in the final row.
};

void check_known_query(const Fixture& f, std::span<const Predicate> predicates,
                       const std::vector<ScoredRecord>& all_expected) {
    for (const auto k : std::array<std::size_t, 5>{0, 1, 3, 100, maximum_size}) {
        auto expected = all_expected;
        expected.resize(std::min(k, expected.size()));
        exact_results(execute_scalar_reference(f.columns, f.embeddings, f.ids, f.query, predicates, k), expected);
        for (const auto size : batch_sizes) {
            exact_results(execute_batched(f.columns, f.embeddings, f.ids, f.query, predicates, k, size), expected);
            ++differential_checks;
        }
    }
}

void hand_calculated_top_k() {
    const Fixture f;
    check_known_query(f, {}, {{20, 8}, {60, 6}, {70, 2}, {80, 2}, {30, 1}, {40, -1}, {50, -2}, {10, -3}});
}

void selectivity_and_and_predicates() {
    const Fixture f;
    const std::array all{Predicate{0, Comparison::greater_equal, std::int32_t{0}}};
    check_known_query(f, all, {{20, 8}, {60, 6}, {70, 2}, {80, 2}, {30, 1}, {40, -1}, {50, -2}, {10, -3}});
    const std::array none{Predicate{0, Comparison::greater, std::int32_t{7}}};
    check_known_query(f, none, {});
    const std::array pair{Predicate{0, Comparison::greater_equal, std::int32_t{2}},
                          Predicate{2, Comparison::less_equal, 1.5F}};
    check_known_query(f, pair, {{20, 8}, {60, 6}, {70, 2}, {50, -2}});
    const std::array triple{pair[0], pair[1],
        Predicate{1, Comparison::not_equal, std::int64_t{9'007'199'254'740'996LL}}};
    check_known_query(f, triple, {{20, 8}, {70, 2}, {50, -2}});
}

void every_comparison_and_type() {
    const ColumnBatch columns({std::vector<std::int32_t>{-2, 0, 2},
        std::vector<std::int64_t>{9'007'199'254'740'992LL, 9'007'199'254'740'993LL, 9'007'199'254'740'994LL},
        std::vector<float>{-0.5F, 0.5F, 1.5F}});
    const EmbeddingBuffer embeddings(3, 1, {1, 1, 1});
    const std::array<RecordID, 3> ids{30, 10, 20};
    const std::array query{1.0F};
    const std::array<NumericConstant, 3> constants{std::int32_t{0}, std::int64_t{9'007'199'254'740'993LL}, 0.5F};
    const std::array cases{
        std::pair{Comparison::less, std::vector<ScoredRecord>{{30, 1}}},
        std::pair{Comparison::less_equal, std::vector<ScoredRecord>{{10, 1}, {30, 1}}},
        std::pair{Comparison::greater, std::vector<ScoredRecord>{{20, 1}}},
        std::pair{Comparison::greater_equal, std::vector<ScoredRecord>{{10, 1}, {20, 1}}},
        std::pair{Comparison::equal, std::vector<ScoredRecord>{{10, 1}}},
        std::pair{Comparison::not_equal, std::vector<ScoredRecord>{{20, 1}, {30, 1}}},
    };
    for (std::size_t column = 0; column < constants.size(); ++column) {
        for (const auto& [op, expected] : cases) {
            const std::array predicates{Predicate{column, op, constants[column]}};
            exact_results(execute_scalar_reference(columns, embeddings, ids, query, predicates, 10), expected);
            for (const auto size : batch_sizes) {
                exact_results(execute_batched(columns, embeddings, ids, query, predicates, 10, size), expected);
                ++differential_checks;
            }
        }
    }
}

void empty_and_single_row() {
    const ColumnBatch no_columns({});
    const ColumnBatch typed_empty({std::vector<float>{}});
    const EmbeddingBuffer empty_embeddings(0, 2, {});
    const std::array query{2.0F, -1.0F};
    const std::array predicate{Predicate{0, Comparison::equal, 1.0F}};
    const ColumnBatch single({std::vector<float>{1}});
    const EmbeddingBuffer single_embedding(1, 2, {4, 3});
    const std::array<RecordID, 1> id{123};
    for (const auto k : std::array<std::size_t, 3>{0, 1, 20}) {
        compare_paths(no_columns, empty_embeddings, {}, query, {}, k);
        compare_paths(typed_empty, empty_embeddings, {}, query, predicate, k);
        compare_paths(single, single_embedding, id, query, predicate, k);
    }
    exact_results(execute_batched(single, single_embedding, id, query, predicate, 20, 3), {{123, 5}});
    reject_in_both<std::invalid_argument>(no_columns, empty_embeddings, {}, {}, {}, 0);
    reject_in_both<std::out_of_range>(no_columns, empty_embeddings, {}, query, predicate, 0);
    const std::array wrong_type{Predicate{0, Comparison::equal, std::int32_t{1}}};
    reject_in_both<std::invalid_argument>(typed_empty, empty_embeddings, {}, query, wrong_type, 0);
}

void every_row_once() {
    for (const auto rows : std::array<std::size_t, 8>{0, 1, 2, 6, 7, 16, 17, 37}) {
        std::vector<std::int32_t> values(rows, 0);
        std::vector<float> embeddings;
        std::vector<RecordID> ids;
        std::vector<ScoredRecord> expected;
        for (std::size_t row = 0; row < rows; ++row) {
            const float score = static_cast<float>(row) - 18.0F;
            const RecordID id = 1'000 + static_cast<RecordID>((row * 13) % 41);
            embeddings.push_back(score);
            ids.push_back(id);
            expected.push_back({id, score});
        }
        // Query is [1], so scores are the embedding values themselves, in reverse
        // row order. All 37 possible IDs are distinct; no filter can conceal gaps.
        std::reverse(expected.begin(), expected.end());
        const ColumnBatch columns({std::move(values)});
        const EmbeddingBuffer buffer(rows, 1, std::move(embeddings));
        const std::array query{1.0F};
        for (const auto size : batch_sizes) {
            exact_results(execute_batched(columns, buffer, ids, query, {}, maximum_size, size), expected);
        }
        compare_paths(columns, buffer, ids, query, {}, maximum_size);
    }
}

void cross_batch_ties() {
    const ColumnBatch columns({std::vector<std::int32_t>{0, 0, 0, 0, 0}});
    const EmbeddingBuffer embeddings(5, 2, {1, 2, 3, 0, 2, 1, 0, 3, -1, 4});
    const std::array<RecordID, 5> ids{90, std::numeric_limits<RecordID>::max(), 50, 0, 10};
    const std::array query{1.0F, 1.0F};
    for (const auto size : batch_sizes) {
        exact_results(execute_batched(columns, embeddings, ids, query, {}, 3, size), {{0, 3}, {10, 3}, {50, 3}});
    }
    compare_paths(columns, embeddings, ids, query, {}, 10);
    const EmbeddingBuffer zeros(5, 1, {-std::numeric_limits<float>::denorm_min(), 0, -0.0F, 0, 0});
    const std::array half{0.5F};
    compare_paths(columns, zeros, ids, half, {}, 5);
}

void invalid_batch_size() {
    const Fixture f;
    const ColumnBatch empty({});
    const EmbeddingBuffer embeddings(0, 2, {});
    for (const auto k : std::array<std::size_t, 2>{0, 2}) {
        expect_throw<std::invalid_argument>([&] {
            (void)execute_batched(f.columns, f.embeddings, f.ids, f.query, {}, k, 0);
        });
        expect_throw<std::invalid_argument>([&] {
            (void)execute_batched(empty, embeddings, {}, f.query, {}, k, 0);
        });
    }
}

void malformed_inputs_and_predicates() {
    const Fixture f;
    auto duplicate_ids = f.ids;
    duplicate_ids.back() = duplicate_ids.front(); // Duplicates span batch boundaries.
    const EmbeddingBuffer wrong_rows(1, 2, {1, 2});
    const std::array short_query{1.0F};
    const std::array long_query{1.0F, 1.0F, 1.0F};
    const std::array impossible{Predicate{0, Comparison::greater, std::int32_t{100}}};
    const std::array wrong_types{Predicate{0, Comparison::equal, 1.0F},
        Predicate{0, Comparison::equal, std::int64_t{1}}, Predicate{1, Comparison::equal, std::int32_t{1}},
        Predicate{1, Comparison::equal, 1.0F}, Predicate{2, Comparison::equal, std::int32_t{1}},
        Predicate{2, Comparison::equal, std::int64_t{1}}};
    for (const auto k : std::array<std::size_t, 2>{0, 2}) {
        reject_in_both<std::invalid_argument>(f.columns, f.embeddings, duplicate_ids, f.query, {}, k);
        reject_in_both<std::invalid_argument>(f.columns, f.embeddings, duplicate_ids, f.query, impossible, k);
        reject_in_both<std::invalid_argument>(f.columns, wrong_rows, f.ids, f.query, {}, k);
        reject_in_both<std::invalid_argument>(f.columns, f.embeddings, std::span(f.ids).first(7), f.query, {}, k);
        reject_in_both<std::invalid_argument>(f.columns, f.embeddings, {}, f.query, {}, k);
        reject_in_both<std::invalid_argument>(f.columns, f.embeddings, f.ids, short_query, {}, k);
        reject_in_both<std::invalid_argument>(f.columns, f.embeddings, f.ids, long_query, {}, k);
        for (const auto& predicate : wrong_types) {
            reject_in_both<std::invalid_argument>(f.columns, f.embeddings, f.ids, f.query, std::span(&predicate, 1), k);
        }
        const std::array bad_index{Predicate{maximum_size, Comparison::equal, 0.0F}};
        reject_in_both<std::out_of_range>(f.columns, f.embeddings, f.ids, f.query, bad_index, k);
        const std::array bad_op{Predicate{0, static_cast<Comparison>(99), std::int32_t{0}}};
        reject_in_both<std::invalid_argument>(f.columns, f.embeddings, f.ids, f.query, bad_op, k);
        const std::array unreachable{impossible[0], wrong_types[0]};
        reject_in_both<std::invalid_argument>(f.columns, f.embeddings, f.ids, f.query, unreachable, k);
    }
}

void non_finite_inputs() {
    const ColumnBatch columns({std::vector<std::int32_t>{1, 2, 3}, std::vector<float>{1, 2, 3}});
    const EmbeddingBuffer embeddings(3, 2, {1, 2, 3, 4, 5, 6});
    const std::array<RecordID, 3> ids{30, 20, 10};
    const std::array query{1.0F, 1.0F};
    const std::array none{Predicate{0, Comparison::less, std::int32_t{0}}};
    for (const auto bad : std::array{std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        const ColumnBatch bad_columns({std::vector<std::int32_t>{1, 2, 3}, std::vector<float>{1, 2, bad}});
        const EmbeddingBuffer bad_embeddings(3, 2, {1, 2, 3, 4, 5, bad});
        const std::array bad_query{1.0F, bad};
        const std::array bad_predicate{Predicate{1, Comparison::equal, bad}};
        for (const auto k : std::array<std::size_t, 2>{0, 2}) {
            reject_in_both<std::invalid_argument>(bad_columns, embeddings, ids, query, none, k);
            reject_in_both<std::invalid_argument>(columns, bad_embeddings, ids, query, none, k);
            reject_in_both<std::invalid_argument>(columns, embeddings, ids, bad_query, none, k);
            reject_in_both<std::invalid_argument>(columns, embeddings, ids, query, bad_predicate, k);
        }
    }
}

void score_overflow_and_zero_k_policy() {
    const ColumnBatch columns({std::vector<std::int32_t>{0, 1}});
    const std::array<RecordID, 2> ids{20, 10};
    constexpr float maximum = std::numeric_limits<float>::max();
    const std::array twice{2.0F};
    const EmbeddingBuffer positive(2, 1, {1, maximum});
    const EmbeddingBuffer negative(2, 1, {1, -maximum});
    reject_in_both<std::overflow_error>(columns, positive, ids, twice, {}, 1);
    reject_in_both<std::overflow_error>(columns, negative, ids, twice, {}, 1);
    const EmbeddingBuffer sum(2, 2, {1, 1, maximum, maximum});
    const std::array ones{1.0F, 1.0F};
    reject_in_both<std::overflow_error>(columns, sum, ids, ones, {}, 1);
    compare_paths(columns, sum, ids, ones, {}, 0); // Finite inputs, no scoring at K=0.
    const std::array first_only{Predicate{0, Comparison::equal, std::int32_t{0}}};
    compare_paths(columns, sum, ids, ones, first_only, 2); // Filtered overflow is never computed.
    for (const auto size : batch_sizes) {
        exact_results(execute_batched(columns, sum, ids, ones, first_only, 2, size), {{20, 2}});
    }
}

void float32_fma_contract() {
    const ColumnBatch columns({std::vector<std::int32_t>{0, 0}});
    const std::array<RecordID, 2> ids{20, 10};
    const EmbeddingBuffer cancellation(2, 3, {16'777'216.0F, 1, -16'777'216.0F, 0, 0, 1});
    const std::array ones{1.0F, 1.0F, 1.0F};
    compare_paths(columns, cancellation, ids, ones, {}, 2);
    exact_results(execute_batched(columns, cancellation, ids, ones, {}, 2, 1), {{10, 1}, {20, 0}});
    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    const EmbeddingBuffer fused(2, 2, {-1.0F, 1.0F + epsilon, 0, 0});
    const std::array query{1.0F, 1.0F - epsilon};
    compare_paths(columns, fused, ids, query, {}, 2);
    // (1+2^-23)(1-2^-23)-1 = -2^-46 under the specified FMA policy.
    exact_results(execute_batched(columns, fused, ids, query, {}, 2, 1), {{10, 0}, {20, -0x1p-46F}});
}

void fixed_seed_randomized_differential() {
    std::mt19937 random(0x56463033U);
    constexpr std::int64_t base = 9'007'199'254'740'992LL;
    constexpr std::array ops{Comparison::less, Comparison::less_equal, Comparison::greater,
                             Comparison::greater_equal, Comparison::equal, Comparison::not_equal};
    for (std::size_t example = 0; example < 48; ++example) {
        const std::size_t rows = example % 7 == 0 ? 0 : random() % 41;
        const std::size_t dimension = 1 + random() % 7;
        std::vector<std::int32_t> small;
        std::vector<std::int64_t> large;
        std::vector<float> real;
        std::vector<float> embeddings;
        std::vector<float> query;
        std::vector<RecordID> ids;
        for (std::size_t row = 0; row < rows; ++row) {
            small.push_back(static_cast<std::int32_t>(random() % 11) - 5);
            large.push_back(base + static_cast<std::int64_t>(random() % 11));
            real.push_back(static_cast<float>(static_cast<int>(random() % 21) - 10) / 4.0F);
            ids.push_back(std::numeric_limits<RecordID>::max() - static_cast<RecordID>(row * 101));
            for (std::size_t feature = 0; feature < dimension; ++feature) {
                embeddings.push_back(static_cast<float>(static_cast<int>(random() % 65) - 32) / 11.0F);
            }
        }
        for (std::size_t feature = 0; feature < dimension; ++feature) {
            query.push_back(static_cast<float>(static_cast<int>(random() % 33) - 16) / 7.0F);
        }
        std::shuffle(ids.begin(), ids.end(), random);
        std::vector<ColumnBatch::Column> storage;
        storage.emplace_back(std::move(small));
        storage.emplace_back(std::move(large));
        storage.emplace_back(std::move(real));
        const ColumnBatch columns(std::move(storage));
        const EmbeddingBuffer buffer(rows, dimension, std::move(embeddings));
        const std::array<std::vector<Predicate>, 4> predicate_sets{
            std::vector<Predicate>{},
            std::vector<Predicate>{{0, Comparison::greater_equal, std::int32_t{0}}},
            std::vector<Predicate>{{1, Comparison::not_equal, base + 1}, {2, Comparison::less, 1.0F}},
            std::vector<Predicate>{{0, ops[random() % ops.size()], std::int32_t{2}},
                {1, ops[random() % ops.size()], base + 5}, {2, ops[random() % ops.size()], -0.5F}},
        };
        for (const auto& predicates : predicate_sets) {
            for (const auto k : std::array<std::size_t, 4>{0, 1, rows / 2 + 1, rows + 5}) {
                compare_paths(columns, buffer, ids, query, predicates, k);
            }
        }
    }
}

void results_outlive_inputs() {
    const auto result = [] {
        const Fixture f;
        return execute_batched(f.columns, f.embeddings, f.ids, f.query, {}, 2, 3);
    }();
    exact_results(result, {{20, 8}, {60, 6}});
    const Fixture f;
    const std::array reverse{-2.0F, 1.0F};
    compare_paths(f.columns, f.embeddings, f.ids, reverse, {}, 8);
    compare_paths(f.columns, f.embeddings, f.ids, f.query, {}, 8);
    check(f.columns.column<std::int32_t>(0)[7] == 7 && f.embeddings.row(7)[0] == 4,
          "execution leaves input storage unchanged");
}

} // namespace

int main() {
    const std::array tests{
        std::pair{"hand-calculated global Top-K", hand_calculated_top_k},
        std::pair{"zero, partial and full selectivity; AND predicates", selectivity_and_and_predicates},
        std::pair{"every comparison and numeric type", every_comparison_and_type},
        std::pair{"empty and single-row datasets", empty_and_single_row},
        std::pair{"all rows exactly once across batch boundaries", every_row_once},
        std::pair{"cross-batch ties and signed-zero score bits", cross_batch_ties},
        std::pair{"zero batch size", invalid_batch_size},
        std::pair{"malformed inputs and predicate validation", malformed_inputs_and_predicates},
        std::pair{"non-finite inputs, even at K=0 or no matches", non_finite_inputs},
        std::pair{"score overflow and K=0 policy", score_overflow_and_zero_k_policy},
        std::pair{"sequential float32 FMA scoring", float32_fma_contract},
        std::pair{"fixed-seed randomized exact differential", fixed_seed_randomized_differential},
        std::pair{"result ownership and independent queries", results_outlive_inputs},
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
    std::cout << (tests.size() - failed) << '/' << tests.size() << " test groups passed; "
              << differential_checks << " exact result comparisons\n";
    return failed == 0 ? 0 : 1;
}
