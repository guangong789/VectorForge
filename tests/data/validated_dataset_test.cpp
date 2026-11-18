#include "vectorforge/validated_dataset.hpp"

#include "vectorforge/batch_view.hpp"
#include "vectorforge/batched_executor.hpp"
#include "vectorforge/scalar_reference.hpp"

#include <array>
#include <bit>
#include <iostream>
#include <limits>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using namespace vectorforge;
constexpr auto maximum = std::numeric_limits<std::size_t>::max();

static_assert(!std::is_default_constructible_v<ValidatedDataset>);
static_assert(!std::is_copy_constructible_v<ValidatedDataset>);
static_assert(std::is_nothrow_move_constructible_v<ValidatedDataset>);
static_assert(!std::is_move_assignable_v<ValidatedDataset>);
static_assert(std::same_as<decltype(std::declval<ValidatedDataset&>().columns()), const ColumnBatch&>);
static_assert(std::same_as<decltype(std::declval<ValidatedDataset&>().embeddings()), const EmbeddingBuffer&>);
static_assert(std::same_as<decltype(std::declval<ValidatedDataset&>().record_ids()), std::span<const RecordID>>);
static_assert(!std::is_assignable_v<decltype(std::declval<ValidatedDataset&>().record_ids()[0]), RecordID>);
static_assert(!std::is_assignable_v<decltype(std::declval<ValidatedDataset&>().columns().column<float>(0)[0]), float>);
static_assert(!std::is_assignable_v<decltype(std::declval<ValidatedDataset&>().embeddings().row(0)[0]), float>);
static_assert(!std::is_constructible_v<ColumnBatch, decltype(std::declval<ValidatedDataset&>().columns())>);
template <typename T>
concept CanBorrowDataset = requires(T&& value) { std::forward<T>(value).record_ids(); };
static_assert(CanBorrowDataset<const ValidatedDataset&> && !CanBorrowDataset<ValidatedDataset>);

void check(bool condition, const char* message,
           const std::source_location location = std::source_location::current()) {
    if (!condition) {
        throw std::runtime_error(std::string(location.file_name()) + ":" +
                                 std::to_string(location.line()) + ": " + message);
    }
}

template <typename Error, typename Function>
void expect_throw(Function&& function) {
    try { std::forward<Function>(function)(); }
    catch (const Error&) { return; }
    throw std::runtime_error("expected exception was not thrown");
}

void exact(std::span<const ScoredRecord> actual, std::span<const ScoredRecord> expected) {
    check(actual.size() == expected.size(), "result count");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        check(actual[i].id == expected[i].id, "ID or deterministic order");
        check(std::bit_cast<std::uint32_t>(actual[i].score) == std::bit_cast<std::uint32_t>(expected[i].score),
              "exact float32 score bits");
    }
}

struct Fixture {
    ColumnBatch columns{{std::vector<std::int32_t>{10, 20, 30, 20, 40},
        std::vector<std::int64_t>{9'007'199'254'740'993LL, 9'007'199'254'740'992LL, 9'007'199'254'740'994LL,
                                 9'007'199'254'740'995LL, 9'007'199'254'740'996LL},
        std::vector<float>{0.5F, 1.5F, -1, 2, 3}}};
    EmbeddingBuffer embeddings{5, 3, {1, 2, 0, 2, 0, 1, -1, 1, 3, 0, 4, -1, 3, -2, 2}};
    std::vector<RecordID> ids{42, 7, 90, 11, 3};
    std::array<float, 3> query{2, -1, 1};
    ValidatedDataset admit() const { return ValidatedDataset(columns, embeddings, ids); }
};

void admission_and_snapshot_ownership() {
    Fixture f;
    const auto dataset = f.admit();
    check(dataset.columns().row_count() == 5 && dataset.embeddings().row_count() == 5, "admitted row shape");
    check(dataset.record_ids().data() != f.ids.data(), "IDs are sealed by copying, not a borrowed const span");
    check(dataset.columns().column<std::int32_t>(0).data() != f.columns.column<std::int32_t>(0).data(),
          "numeric payload is detached from any preexisting source aliases");
    check(dataset.columns().column<std::int64_t>(1).data() != f.columns.column<std::int64_t>(1).data(),
          "int64 payload has independent storage");
    check(dataset.columns().column<float>(2).data() != f.columns.column<float>(2).data(), "float payload detached");
    check(dataset.embeddings().row(0).data() != f.embeddings.row(0).data(), "embedding payload detached");
    auto* retained_id_alias = f.ids.data();
    retained_id_alias[0] = retained_id_alias[1];
    f.ids.clear();
    f.ids.shrink_to_fit();
    check(dataset.record_ids()[0] == 42 && dataset.record_ids()[1] == 7 && dataset.record_ids().size() == 5,
          "mutating/resizing source IDs cannot invalidate admitted invariants");
    const std::vector<ScoredRecord> expected{{3, 10}, {7, 5}, {42, 0}, {90, 0}, {11, -5}};
    exact(execute_scalar_reference(dataset, f.query, {}, 100), expected);
    exact(execute_batched(dataset, f.query, {}, 100, 2), expected);
    const auto independent = [] { const Fixture source; return source.admit(); }();
    exact(execute_batched(independent, f.query, {}, 100, 3), expected);
}

void duplicate_and_alignment_rejection() {
    Fixture f;
    f.ids.back() = f.ids.front();
    expect_throw<std::invalid_argument>([&] { (void)f.admit(); });
    // Legacy entry points must still reject duplicates, also for K=0/no matches.
    const std::array none{Predicate{0, Comparison::greater, std::int32_t{100}}};
    for (const auto k : std::array<std::size_t, 2>{0, 2}) {
        expect_throw<std::invalid_argument>([&] {
            (void)execute_scalar_reference(f.columns, f.embeddings, f.ids, f.query, none, k);
        });
        expect_throw<std::invalid_argument>([&] {
            (void)execute_batched(f.columns, f.embeddings, f.ids, f.query, none, k, 2);
        });
    }
    f.ids.back() = 3;
    const EmbeddingBuffer wrong_rows(1, 3, {1, 2, 3});
    expect_throw<std::invalid_argument>([&] { const ValidatedDataset bad(f.columns, wrong_rows, f.ids); });
    expect_throw<std::invalid_argument>([&] {
        const ValidatedDataset bad(f.columns, f.embeddings, std::span(f.ids).first(4));
    });
    auto long_ids = f.ids;
    long_ids.push_back(100);
    expect_throw<std::invalid_argument>([&] { const ValidatedDataset bad(f.columns, f.embeddings, long_ids); });
    const auto repaired = f.admit();
    check(repaired.record_ids().size() == 5, "rejected admissions do not consume or modify inputs");
}

void move_and_view_lifetimes() {
    std::optional<ValidatedDataset> destination;
    std::optional<BatchView> view;
    std::span<const RecordID> ids;
    const ColumnBatch* original_columns = nullptr;
    {
        const Fixture f;
        auto source = f.admit();
        ids = source.record_ids();
        original_columns = &source.columns();
        view.emplace(source.columns(), source.embeddings(), 1, 3);
        destination.emplace(std::move(source));
        check(&destination->columns() == original_columns && destination->record_ids().data() == ids.data(),
              "dataset moves transfer stable sealed storage without copying");
        expect_throw<std::logic_error>([&] { (void)source.columns(); });
        expect_throw<std::logic_error>([&] { (void)execute_scalar_reference(source, f.query, {}, 0); });
        expect_throw<std::logic_error>([&] { (void)execute_batched(source, f.query, {}, 0, 1); });
        expect_throw<std::invalid_argument>([&] {
            (void)execute_batched(source, f.query, {}, 0, 0); // Reject the batch size before accessing the snapshot.
        });
    }
    check(ids[0] == 42 && ids[4] == 3, "ID view survives source-owner destruction after move");
    check(view->global_row(0) == 1 && view->column<std::int32_t>(0)[0] == 20 && view->embedding_row(0)[2] == 1,
          "BatchView remains aligned when the dataset owner moves");
}

void reused_queries_and_exact_equivalence() {
    const Fixture f;
    const auto dataset = f.admit();
    const auto* stable_ids = dataset.record_ids().data();
    const std::array<std::vector<Predicate>, 6> filters{
        std::vector<Predicate>{},
        std::vector<Predicate>{{0, Comparison::greater_equal, std::int32_t{20}}, {2, Comparison::less_equal, 2.0F}},
        std::vector<Predicate>{{1, Comparison::equal, std::int64_t{9'007'199'254'740'993LL}}},
        std::vector<Predicate>{{0, Comparison::less, std::int32_t{20}}},
        std::vector<Predicate>{{2, Comparison::greater, 1.5F}},
        std::vector<Predicate>{{0, Comparison::not_equal, std::int32_t{20}}},
    };
    for (const auto query : std::array{f.query, std::array{-2.0F, 1.0F, -1.0F}, std::array{0.0F, 0.0F, 0.0F}}) {
        for (const auto& predicates : filters) {
            for (const auto k : std::array<std::size_t, 5>{0, 1, 2, 5, maximum}) {
                const auto expected = execute_scalar_reference(f.columns, f.embeddings, f.ids, query, predicates, k);
                exact(execute_scalar_reference(dataset, query, predicates, k), expected);
                for (const auto batch_size : std::array<std::size_t, 5>{1, 2, 3, 8, maximum}) {
                    exact(execute_batched(dataset, query, predicates, k, batch_size), expected);
                }
            }
        }
    }
    check(dataset.record_ids().data() == stable_ids, "queries retain the same immutable dataset storage");
}

template <typename Error>
void reject_query(const ValidatedDataset& dataset, std::span<const float> query,
                  std::span<const Predicate> predicates, std::size_t k) {
    expect_throw<Error>([&] { (void)execute_scalar_reference(dataset, query, predicates, k); });
    expect_throw<Error>([&] { (void)execute_batched(dataset, query, predicates, k, 2); });
    expect_throw<Error>([&] {
        (void)execute_scalar_reference(dataset.columns(), dataset.embeddings(), dataset.record_ids(), query, predicates, k);
    });
}

void query_validation_including_zero_k() {
    const Fixture f;
    const auto dataset = f.admit();
    const std::array short_query{1.0F, 2.0F};
    const std::array long_query{1.0F, 2.0F, 3.0F, 4.0F};
    for (const auto k : std::array<std::size_t, 2>{0, 2}) {
        reject_query<std::invalid_argument>(dataset, {}, {}, k);
        reject_query<std::invalid_argument>(dataset, short_query, {}, k);
        reject_query<std::invalid_argument>(dataset, long_query, {}, k);
        for (const auto bad : std::array{std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
            const std::array bad_query{1.0F, 2.0F, bad};
            reject_query<std::invalid_argument>(dataset, bad_query, {}, k);
            const std::array bad_constant{Predicate{2, Comparison::less, bad}};
            reject_query<std::invalid_argument>(dataset, f.query, bad_constant, k);
        }
        const std::array bad_type{Predicate{0, Comparison::equal, 20.0F}};
        reject_query<std::invalid_argument>(dataset, f.query, bad_type, k);
        const std::array bad_operator{Predicate{0, static_cast<Comparison>(99), std::int32_t{20}}};
        reject_query<std::invalid_argument>(dataset, f.query, bad_operator, k);
        const std::array bad_column{Predicate{maximum, Comparison::equal, 1.0F}};
        reject_query<std::out_of_range>(dataset, f.query, bad_column, k);
        const std::array bad_operator_and_column{Predicate{maximum, bad_operator[0].comparison, 1.0F}};
        const std::array bad_column_then_operator{bad_column[0], bad_operator[0]};
        expect_throw<std::invalid_argument>([&] {
            (void)execute_batched(dataset, short_query, bad_column, k, 2);
        });
        expect_throw<std::invalid_argument>([&] {
            (void)execute_batched(dataset, f.query, bad_operator_and_column, k, 2);
        });
        expect_throw<std::out_of_range>([&] {
            (void)execute_batched(dataset, f.query, bad_column_then_operator, k, 2);
        });
        const std::array unreachable{Predicate{0, Comparison::greater, std::int32_t{100}}, bad_type[0]};
        reject_query<std::invalid_argument>(dataset, f.query, unreachable, k);
        expect_throw<std::invalid_argument>([&] { (void)execute_batched(dataset, f.query, {}, k, 0); });
    }
}

void admission_finiteness_and_score_contract() {
    const std::array<RecordID, 1> ids{9};
    const ColumnBatch finite_columns({std::vector<float>{1}});
    const EmbeddingBuffer finite_embeddings(1, 1, {1});
    for (const auto bad : std::array{std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        const ColumnBatch bad_columns({std::vector<float>{bad}});
        const EmbeddingBuffer bad_embeddings(1, 1, {bad});
        expect_throw<std::invalid_argument>([&] { const ValidatedDataset invalid(bad_columns, finite_embeddings, ids); });
        expect_throw<std::invalid_argument>([&] { const ValidatedDataset invalid(finite_columns, bad_embeddings, ids); });
    }
    const EmbeddingBuffer huge(1, 1, {std::numeric_limits<float>::max()});
    const ValidatedDataset overflow(finite_columns, huge, ids); // Finite data are admissible.
    const std::array twice{2.0F};
    exact(execute_scalar_reference(overflow, twice, {}, 0), {});
    exact(execute_batched(overflow, twice, {}, 0, 1), {});
    reject_query<std::overflow_error>(overflow, twice, {}, 1);
    constexpr auto epsilon = std::numeric_limits<float>::epsilon();
    const EmbeddingBuffer fused(1, 2, {-1, 1 + epsilon});
    const ValidatedDataset rounding(finite_columns, fused, ids);
    const std::array query{1.0F, 1 - epsilon};
    const std::vector<ScoredRecord> expected{{9, -0x1p-46F}};
    exact(execute_scalar_reference(rounding, query, {}, 1), expected);
    exact(execute_batched(rounding, query, {}, 1, 1), expected);
}

void empty_and_single_row() {
    const ColumnBatch columns({});
    const EmbeddingBuffer embeddings(0, 2, {});
    const ValidatedDataset empty(columns, embeddings, {});
    const std::array query{2.0F, -1.0F};
    for (const auto k : std::array<std::size_t, 3>{0, 1, maximum}) {
        exact(execute_scalar_reference(empty, query, {}, k), {});
        exact(execute_batched(empty, query, {}, k, 3), {});
    }
    reject_query<std::invalid_argument>(empty, {}, {}, 0);
    const ColumnBatch typed({std::vector<float>{}});
    const ValidatedDataset typed_empty(typed, embeddings, {});
    const std::array predicate{Predicate{0, Comparison::equal, 1.0F}};
    exact(execute_batched(typed_empty, query, predicate, 1, 1), {});
    const std::array wrong_type{Predicate{0, Comparison::equal, std::int32_t{1}}};
    reject_query<std::invalid_argument>(typed_empty, query, wrong_type, 0);
    const ColumnBatch single_column({std::vector<float>{1}});
    const EmbeddingBuffer single_embedding(1, 2, {3, 4});
    const std::array<RecordID, 1> id{std::numeric_limits<RecordID>::max()};
    const ValidatedDataset single(single_column, single_embedding, id);
    const std::vector<ScoredRecord> expected{{id[0], 2}};
    exact(execute_scalar_reference(single, query, predicate, maximum), expected);
    exact(execute_batched(single, query, predicate, maximum, maximum), expected);
}

} // namespace

int main() {
    const std::array tests{
        std::pair{"sealed snapshot and source aliases", admission_and_snapshot_ownership},
        std::pair{"duplicate IDs, alignment and legacy behavior", duplicate_and_alignment_rejection},
        std::pair{"move ownership and borrowed-view lifetime", move_and_view_lifetimes},
        std::pair{"reused trusted queries and exact equivalence", reused_queries_and_exact_equivalence},
        std::pair{"query validation including K=0", query_validation_including_zero_k},
        std::pair{"admission finiteness and FMA/overflow contract", admission_finiteness_and_score_contract},
        std::pair{"empty and single-row datasets", empty_and_single_row},
    };
    std::size_t failed = 0;
    for (const auto& [name, test] : tests) {
        try { test(); std::cout << "PASS: " << name << '\n'; }
        catch (const std::exception& error) { ++failed; std::cerr << "FAIL: " << name << ": " << error.what() << '\n'; }
        catch (...) { ++failed; std::cerr << "FAIL: " << name << ": unknown exception\n"; }
    }
    std::cout << (tests.size() - failed) << '/' << tests.size() << " test groups passed\n";
    return failed == 0 ? 0 : 1;
}
