#include "vectorforge/filtered_count.hpp"
#include "vectorforge/filtered_sum.hpp"
#include "vectorforge/column_batch.hpp"
#include "vectorforge/embedding_buffer.hpp"
#include "vectorforge/validated_dataset.hpp"

#include "execution/numeric_column_validation.hpp"
#include "execution/predicate_evaluation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>
#include <string_view>
#include <utility>

// Narrow, single-threaded new/new[] probe, matching the SUM test convention.
namespace allocations {
bool active = false;
std::size_t calls = 0;
#if defined(__GNUC__)
[[gnu::noinline]]
#endif
void release(void* p) noexcept { std::free(p); }
}
void* operator new(std::size_t bytes) {
    if (allocations::active) { ++allocations::calls; }
    if (void* p = std::malloc(bytes == 0 ? 1 : bytes)) { return p; }
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* p) noexcept { allocations::release(p); }
void operator delete[](void* p) noexcept { allocations::release(p); }
void operator delete(void* p, std::size_t) noexcept { allocations::release(p); }
void operator delete[](void* p, std::size_t) noexcept { allocations::release(p); }

namespace {
using namespace vectorforge;
constexpr auto maximum = std::numeric_limits<std::size_t>::max();
constexpr std::int64_t big = 9'007'199'254'740'993LL;
constexpr std::array<std::size_t, 7> batches{1, 2, 4, 7, 32, 999, maximum};
constexpr std::array operators{Comparison::less, Comparison::less_equal, Comparison::greater,
    Comparison::greater_equal, Comparison::equal, Comparison::not_equal};
std::size_t checks = 0, rejections = 0, compatible = 0;

void check(bool value, const char* message) { if (!value) { throw std::runtime_error(message); } }
struct Probe {
    Probe() { allocations::calls = 0; allocations::active = true; }
    ~Probe() { allocations::active = false; }
};
void calibrate() {
    const Probe probe;
    void* p = ::operator new(8);
    const auto calls = allocations::calls;
    ::operator delete(p);
    check(calls == 1, "allocation probe must observe explicit operator new");
}

// Independent raw COUNT oracle. No production validation, predicate or COUNT
// helpers; column access is the data foundation. Validates all shapes BEFORE
// inspecting float payloads, then interprets each predicate and counts by row.
std::size_t oracle(const ColumnBatch& columns, std::span<const Predicate> predicates,
                   std::size_t batch_size) {
    if (batch_size == 0) { throw std::invalid_argument("oracle batch"); }
    for (std::size_t column = 0; column < columns.column_count(); ++column) {
        std::visit([&](auto values) {
            if (values.size() != columns.row_count()) { throw std::invalid_argument("oracle shape"); }
        }, columns.column_view(column));
    }
    for (std::size_t column = 0; column < columns.column_count(); ++column) {
        const auto view = columns.column_view(column);
        if (const auto* values = std::get_if<std::span<const float>>(&view)) {
            for (const auto value : *values) {
                if (!std::isfinite(value)) { throw std::invalid_argument("oracle float data"); }
            }
        }
    }
    for (const auto& predicate : predicates) {
        if (std::find(operators.begin(), operators.end(), predicate.comparison) == operators.end()) {
            throw std::invalid_argument("oracle operator");
        }
        if (columns.column_view(predicate.column_index).index() != predicate.constant.index()) {
            throw std::invalid_argument("oracle constant type");
        }
        if (const auto* value = std::get_if<float>(&predicate.constant); value && !std::isfinite(*value)) {
            throw std::invalid_argument("oracle float constant");
        }
    }
    std::size_t count = 0;
    for (std::size_t row = 0; row < columns.row_count(); ++row) {
        bool selected = true;
        for (const auto& predicate : predicates) {
            const auto matches = std::visit([&]<typename T>(T constant) {
                const auto value = columns.column<T>(predicate.column_index)[row];
                const int order = static_cast<int>(value > constant) - static_cast<int>(value < constant);
                switch (predicate.comparison) {
                case Comparison::less: return order == -1;
                case Comparison::less_equal: return order != 1;
                case Comparison::greater: return order == 1;
                case Comparison::greater_equal: return order != -1;
                case Comparison::equal: return order == 0;
                case Comparison::not_equal: return order != 0;
                }
                throw std::invalid_argument("oracle operator");
            }, predicate.constant);
            if (!matches) { selected = false; break; }
        }
        if (selected) { ++count; }
    }
    return count;
}

void verify(const ColumnBatch& columns, std::span<const Predicate> predicates, std::size_t expected) {
    check(oracle(columns, predicates, 1) == expected, "independent count/hand result");
    for (const auto batch : batches) {
        const Probe probe;
        check(count_where(columns, predicates, batch) == expected, "raw count/batch boundaries");
        check(count_where(columns, predicates, batch) == expected, "raw count repeated consumption");
        check(allocations::calls == 0, "successful raw COUNT allocated");
        ++checks;
    }
}

template <typename Error, typename F> void throws(F function) {
    try { function(); }
    catch (const Error&) { ++rejections; return; }
    throw std::runtime_error("missing expected exception");
}
template <typename Error>
void reject(const ColumnBatch& columns, std::span<const Predicate> predicates, std::size_t batch) {
    throws<Error>([&] { (void)oracle(columns, predicates, batch); });
    throws<Error>([&] { (void)count_where(columns, predicates, batch); });
}
template <typename F> void diagnostic(F function, std::string_view expected) {
    try { function(); }
    catch (const std::invalid_argument& e) {
        check(e.what() == expected, "validation/error ordering changed"); ++rejections; return;
    }
    throw std::runtime_error("missing ordered validation error");
}

void known_counts() {
    // These fixtures and all calls below need only numeric columns: no IDs or embeddings.
    const ColumnBatch columns({std::vector<std::int32_t>{-2, 0, 2, 2, 4, 6},
        std::vector<std::int64_t>{big, big + 1, big + 2, big + 3, big + 4, big + 5},
        std::vector<float>{-1, -0.0F, 0.0F, 0.5F, 1, 2}});
    verify(columns, {}, 6);
    const std::array<NumericConstant, 3> constants{std::int32_t{2}, big + 2, 0.0F};
    const std::array<std::array<std::size_t, 6>, 3> expected{{{2, 4, 2, 4, 2, 4},
        {2, 3, 3, 4, 1, 5}, {1, 3, 3, 5, 2, 4}}};
    for (std::size_t column = 0; column < 3; ++column) {
        for (std::size_t op = 0; op < operators.size(); ++op) {
            const std::array predicates{Predicate{column, operators[op], constants[column]}};
            verify(columns, predicates, expected[column][op]);
        }
    }
    const std::array partial{Predicate{0, Comparison::greater_equal, std::int32_t{2}},
                             Predicate{2, Comparison::less_equal, 1.0F}};
    const std::array three{partial[0], partial[1], Predicate{1, Comparison::not_equal, big + 3}};
    verify(columns, partial, 3); verify(columns, three, 2);
    verify(columns, std::array{Predicate{0, Comparison::less, std::int32_t{-99}}}, 0);
    verify(columns, std::array{Predicate{0, Comparison::greater_equal, std::int32_t{-2}}}, 6);
    const ColumnBatch empty({std::vector<std::int32_t>{}, std::vector<std::int64_t>{}, std::vector<float>{}});
    verify(empty, {}, 0); verify(empty, three, 0);
    const ColumnBatch none({}); verify(none, {}, 0);
    const ColumnBatch float_only({std::vector<float>{-0.0F, 0.0F, 1.0F}});
    verify(float_only, std::array{Predicate{0, Comparison::equal, 0.0F}}, 2);
    const ColumnBatch one({std::vector<std::int64_t>{std::numeric_limits<std::int64_t>::max()}});
    verify(one, {}, 1);
    verify(one, std::array{Predicate{0, Comparison::equal, std::numeric_limits<std::int64_t>::max()}}, 1);
    verify(one, std::array{Predicate{0, Comparison::less, std::numeric_limits<std::int64_t>::min()}}, 0);
    auto moved = ColumnBatch({std::vector<std::int32_t>{1}});
    const auto owner = std::move(moved);
    verify(moved, {}, 0); verify(owner, {}, 1);
    reject<std::out_of_range>(moved, std::array{Predicate{0, Comparison::equal, std::int32_t{1}}}, 1);
}

void invalid_queries() {
    const ColumnBatch columns({std::vector<std::int32_t>{7}, std::vector<std::int64_t>{big}, std::vector<float>{0}});
    const ColumnBatch empty({std::vector<std::int32_t>{}, std::vector<std::int64_t>{}, std::vector<float>{}});
    const std::array bad_index{Predicate{3, Comparison::equal, std::int32_t{0}}};
    const std::array bad_type{Predicate{0, Comparison::equal, std::int64_t{7}}};
    for (const auto* input : {&columns, &empty}) {
        for (const auto batch : batches) {
            reject<std::out_of_range>(*input, bad_index, batch);
            for (const auto bad : std::array{bad_type[0], Predicate{1, Comparison::equal, 1.0F},
                Predicate{2, Comparison::equal, std::int32_t{0}},
                Predicate{0, static_cast<Comparison>(-1), std::int32_t{0}},
                Predicate{0, static_cast<Comparison>(99), std::int32_t{0}},
                Predicate{2, Comparison::equal, std::numeric_limits<float>::quiet_NaN()},
                Predicate{2, Comparison::not_equal, std::numeric_limits<float>::infinity()},
                Predicate{2, Comparison::less, -std::numeric_limits<float>::infinity()}}) {
                reject<std::invalid_argument>(*input, std::array{bad}, batch);
            }
            const std::array none_bad{Predicate{0, Comparison::less, std::int32_t{-99}}, bad_index[0]};
            reject<std::out_of_range>(*input, none_bad, batch);
            reject<std::invalid_argument>(*input, std::array{bad_type[0], bad_index[0]}, batch);
            reject<std::out_of_range>(*input, std::array{bad_index[0], bad_type[0]}, batch);
            reject<std::invalid_argument>(*input,
                std::array{Predicate{maximum, static_cast<Comparison>(99), std::int32_t{0}}}, batch);
        }
        reject<std::invalid_argument>(*input, {}, 0);
        reject<std::invalid_argument>(*input, bad_index, 0);
    }
    for (const auto value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()}) {
        const ColumnBatch data({std::vector<std::int32_t>{7}, std::vector<float>{value}});
        reject<std::invalid_argument>(data, {}, 1); // Unused float, empty-predicate early return.
        reject<std::invalid_argument>(data, bad_index, 1); // Data before predicates.
        reject<std::invalid_argument>(data, std::array{Predicate{0, Comparison::less, std::int32_t{0}}}, 1);
        diagnostic([&] { (void)count_where(data, bad_index, 0); }, "batch size must be positive");
    }
}

void malformed_shapes_and_sum_order() {
    // References to vector elements remain valid when their outer vector is moved.
    // Mutate retained source aliases BEFORE the call, not during a borrowed scan.
    for (const auto length : std::array<std::size_t, 3>{0, 1, 3}) {
        std::vector<ColumnBatch::Column> storage{std::vector<std::int32_t>{1, 2}, std::vector<float>{0, 1}};
        auto& alias = std::get<std::vector<float>>(storage[1]);
        const ColumnBatch data(std::move(storage)); alias.resize(length);
        reject<std::invalid_argument>(data, {}, 1);
        reject<std::invalid_argument>(data, std::array{Predicate{0, Comparison::equal, std::int32_t{1}}}, maximum);
    }
    std::vector<ColumnBatch::Column> empty_storage{std::vector<std::int32_t>{}, std::vector<float>{}};
    auto& empty_alias = std::get<std::vector<float>>(empty_storage[1]);
    const ColumnBatch empty(std::move(empty_storage)); empty_alias.push_back(0);
    reject<std::invalid_argument>(empty, {}, 1); // Shape validation before zero-row return.

    std::vector<ColumnBatch::Column> storage{std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 0},
        std::vector<std::int32_t>{1, 2}};
    auto& alias = std::get<std::vector<std::int32_t>>(storage[1]);
    const ColumnBatch data(std::move(storage)); alias.pop_back();
    diagnostic([&] { (void)count_where(data, {}, 1); }, "numeric column length does not match row count");
    diagnostic([&] { (void)oracle(data, {}, 1); }, "oracle shape");
    diagnostic([&] { (void)count_where(data, {}, 0); }, "batch size must be positive");
    // SUM's published per-column validation precedence and diagnostic stay intact.
    diagnostic([&] { (void)sum_where(data, 1, {}, 1); }, "SUM float columns must be finite");
}

void trusted_compatibility(const ColumnBatch& columns, std::span<const Predicate> predicates) {
    const auto rows = columns.row_count();
    const EmbeddingBuffer embeddings(rows, 1, std::vector<float>(rows, 0));
    std::vector<RecordID> ids(rows);
    for (std::size_t row = 0; row < rows; ++row) { ids[row] = static_cast<RecordID>(rows - row); }
    const ValidatedDataset dataset(columns, embeddings, ids);
    const auto expected = oracle(columns, predicates, 1);
    using Raw = std::size_t (*)(const ColumnBatch&, std::span<const Predicate>, std::size_t);
    using Trusted = std::size_t (*)(const ValidatedDataset&, std::span<const Predicate>, std::size_t);
    const Raw raw = &count_where;
    const Trusted trusted = &count_where; // Original public signature remains selectable.
    for (const auto batch : batches) {
        const Probe probe;
        check(raw(columns, predicates, batch) == expected && trusted(dataset, predicates, batch) == expected,
              "numeric/trusted COUNT equivalence and overload resolution");
        check(allocations::calls == 0, "raw/trusted successful COUNT allocated"); ++compatible;
    }
}

void fixed_seed_differential() {
    std::mt19937_64 random(0x56463109BULL);
    for (std::size_t trial = 0; trial < 24; ++trial) {
        const auto rows = trial == 0 ? std::size_t{0} : static_cast<std::size_t>(random() % 258);
        std::vector<std::int32_t> a(rows);
        std::vector<std::int64_t> b(rows);
        std::vector<float> c(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            a[row] = static_cast<std::int32_t>(random() % 17) - 8;
            b[row] = big + static_cast<std::int64_t>(random() % 17) - 8;
            c[row] = static_cast<float>(static_cast<int>(random() % 17) - 8) / 4.0F;
        }
        const ColumnBatch columns({std::move(a), std::move(b), std::move(c)});
        verify(columns, {}, rows); trusted_compatibility(columns, {});
        for (const auto comparison : operators) {
            const std::array predicates{Predicate{0, comparison, std::int32_t{0}},
                Predicate{1, comparison, big}, Predicate{2, comparison, 0.0F}};
            for (const auto& predicate : predicates) {
                const auto one = std::span(&predicate, 1);
                verify(columns, one, oracle(columns, one, 1));
            }
            verify(columns, predicates, oracle(columns, predicates, 1));
            trusted_compatibility(columns, predicates);
        }
    }
}

// Benchmark-only validation timing. This is deliberately separate from the oracle;
// it times the actual private checks used by raw COUNT, with no matching scan.
std::size_t validation_only(const ColumnBatch& columns, std::span<const Predicate> predicates) {
    for (std::size_t column = 0; column < columns.column_count(); ++column) {
        detail::validate_column_length(columns.column_view(column), columns.row_count());
    }
    for (std::size_t column = 0; column < columns.column_count(); ++column) {
        detail::validate_column_floats(columns.column_view(column), "COUNT float columns must be finite");
    }
    detail::validate_predicates(columns, predicates);
    return columns.row_count();
}

// Opt-in Release workload: vectorforge_numeric_count_tests --workload.
// The public total and validation-only component are timed independently; their
// difference is NOT reported as an isolated predicate latency or a trusted speedup.
void workload() {
    constexpr std::size_t samples = 9, repetitions = 5;
    std::cout << "# deterministic flag=row%100; unused_float=(row%17)/8; warmup=2; samples=9; repetitions=5; alternating order\n"
              << "rows,batch,case,scope,median_us,min_us,max_us,allocations_per_query,count\n";
    using Clock = std::chrono::steady_clock;
    for (const auto rows : std::array<std::size_t, 2>{65'536, 1'048'576}) {
        std::vector<std::int32_t> flags(rows);
        std::vector<float> floats(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            flags[row] = static_cast<std::int32_t>(row % 100);
            floats[row] = static_cast<float>(row % 17) / 8.0F;
        }
        const ColumnBatch columns({std::move(flags), std::move(floats)});
        for (const auto batch : std::array<std::size_t, 2>{32, 1024}) {
            for (const auto percentage : std::array<int, 6>{-1, 0, 1, 50, 90, 100}) {
                const std::array filter{Predicate{0, Comparison::less, static_cast<std::int32_t>(percentage)}};
                const auto predicates = percentage < 0 ? std::span<const Predicate>{} : std::span<const Predicate>{filter};
                const auto expected = oracle(columns, predicates, 1);
                auto query = [&] { return count_where(columns, predicates, batch); };
                auto validation = [&] { return validation_only(columns, predicates); };
                for (int warmup = 0; warmup < 2; ++warmup) {
                    check(query() == expected && validation() == rows, "workload warmup correctness");
                }
                auto measure = [&](auto function, std::size_t expected_count) {
                    std::size_t checksum = 0;
                    double us;
                    {
                        const Probe probe;
                        const auto start = Clock::now();
                        for (std::size_t repeat = 0; repeat < repetitions; ++repeat) { checksum += function(); }
                        const auto finish = Clock::now();
                        us = std::chrono::duration<double, std::micro>(finish - start).count() / static_cast<double>(repetitions);
                    }
                    check(allocations::calls == 0, "workload allocated");
                    check(checksum == expected_count * repetitions, "timed count checksum");
                    return us;
                };
                std::array<double, samples> total{}, validation_times{};
                for (std::size_t sample = 0; sample < samples; ++sample) {
                    if ((sample + static_cast<std::size_t>(percentage + 1)) % 2 == 0) {
                        total[sample] = measure(query, expected); validation_times[sample] = measure(validation, rows);
                    } else {
                        validation_times[sample] = measure(validation, rows); total[sample] = measure(query, expected);
                    }
                }
                auto report = [&](const char* scope, auto times, std::size_t count) {
                    std::sort(times.begin(), times.end());
                    std::cout << rows << ',' << batch << ',';
                    if (percentage < 0) { std::cout << "no_predicates"; } else { std::cout << percentage; }
                    std::cout << ',' << scope << ',' << times[samples / 2] << ',' << times.front() << ','
                              << times.back() << ",0," << count << '\n';
                };
                report("public_total", total, expected); report("validation_only", validation_times, rows);
            }
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        calibrate();
        if (argc == 2 && std::string_view(argv[1]) == "--workload") { workload(); return 0; }
        check(argc == 1, "use no arguments or --workload");
        known_counts(); invalid_queries(); malformed_shapes_and_sum_order(); fixed_seed_differential();
        std::cout << "Numeric COUNT: " << checks << " exact batch checks; " << rejections << " expected rejections; "
                  << compatible << " trusted comparisons; zero successful query allocations\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
