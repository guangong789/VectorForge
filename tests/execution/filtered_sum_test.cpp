#include "vectorforge/filtered_sum.hpp"
#include "vectorforge/column_batch.hpp"

#include "execution/checked_sum.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

// Same narrow malloc/free replacement convention as candidate_scoring_test.
// Only successful query calls are measured, after constructing fixtures/oracles.
namespace allocations {
bool active = false;
std::size_t calls = 0;
// Test replacements pair malloc with free. Keep deallocation out of line so GCC
// does not mistake an inlined free for a mismatched standard delete.
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
constexpr auto low64 = std::numeric_limits<std::int64_t>::min();
constexpr auto high64 = std::numeric_limits<std::int64_t>::max();
constexpr auto low32 = std::numeric_limits<std::int32_t>::min();
constexpr auto high32 = std::numeric_limits<std::int32_t>::max();
constexpr std::int64_t big = 9'007'199'254'740'993LL;
constexpr std::array<std::size_t, 7> batch_sizes{1, 2, 4, 7, 32, 999, maximum};
constexpr std::array comparisons{Comparison::less, Comparison::less_equal,
    Comparison::greater, Comparison::greater_equal, Comparison::equal, Comparison::not_equal};
std::size_t checks = 0, rejected = 0;

struct AllocationProbe {
    AllocationProbe() { allocations::calls = 0; allocations::active = true; }
    ~AllocationProbe() { allocations::active = false; }
};

void check_allocation_probe() {
    const AllocationProbe probe;
    void* p = ::operator new(sizeof(std::int32_t));
    const auto calls = allocations::calls;
    ::operator delete(p);
    if (calls != 1) { throw std::runtime_error("allocation probe did not observe explicit operator new"); }
}
void check(bool condition, const char* message,
           const std::source_location where = std::source_location::current()) {
    if (!condition) {
        throw std::runtime_error(std::string(where.file_name()) + ":" + std::to_string(where.line()) + ": " + message);
    }
}
template <typename Error, typename Function>
void expect_throw(Function&& function) {
    try { function(); }
    catch (const Error&) { ++rejected; return; }
    throw std::runtime_error("expected exception was not thrown");
}

// Independent arithmetic: inspect remaining room from total, rather than using
// the production helper's limit-minus-delta guard. Delta is always an int32.
std::int64_t oracle_add(std::int64_t total, std::int32_t value) {
    const auto delta = static_cast<std::int64_t>(value);
    if ((total >= 0 && delta > 0 && delta > high64 - total) ||
        (total < 0 && delta < 0 && delta < low64 - total)) {
        throw std::overflow_error("oracle int64 overflow");
    }
    return total + delta;
}

// Independent row-by-row oracle: no production predicate/validation helper,
// batching, embedding, RecordID, scorer or checked-add helper is used here.
std::int64_t oracle(const ColumnBatch& columns, std::size_t target_column,
                    std::span<const Predicate> predicates, std::size_t batch_size) {
    if (batch_size == 0) { throw std::invalid_argument("oracle batch size"); }
    for (std::size_t column = 0; column < columns.column_count(); ++column) {
        const auto view = columns.column_view(column);
        std::visit([&](auto values) {
            if (values.size() != columns.row_count()) { throw std::invalid_argument("oracle alignment"); }
        }, view);
        if (const auto* values = std::get_if<std::span<const float>>(&view)) {
            for (const auto value : *values) {
                if (!std::isfinite(value)) { throw std::invalid_argument("oracle float data"); }
            }
        }
    }
    const auto view = columns.column_view(target_column);
    const auto* target = std::get_if<std::span<const std::int32_t>>(&view);
    if (target == nullptr) { throw std::invalid_argument("oracle target type"); }
    for (const auto& predicate : predicates) {
        if (std::find(comparisons.begin(), comparisons.end(), predicate.comparison) == comparisons.end()) {
            throw std::invalid_argument("oracle comparison");
        }
        if (columns.column_view(predicate.column_index).index() != predicate.constant.index()) {
            throw std::invalid_argument("oracle constant type");
        }
        if (const auto* value = std::get_if<float>(&predicate.constant); value && !std::isfinite(*value)) {
            throw std::invalid_argument("oracle float constant");
        }
    }
    std::int64_t total = 0;
    for (std::size_t row = 0; row < columns.row_count(); ++row) {
        bool selected = true;
        for (const auto& predicate : predicates) {
            const bool match = std::visit([&]<typename T>(T constant) {
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
                throw std::invalid_argument("oracle comparison");
            }, predicate.constant);
            if (!match) { selected = false; break; }
        }
        if (selected) { total = oracle_add(total, (*target)[row]); }
    }
    return total;
}

void successful(const ColumnBatch& columns, std::span<const Predicate> predicates, std::int64_t expected) {
    check(oracle(columns, 0, predicates, 1) == expected, "known result agrees with independent oracle");
    const auto* storage = columns.column<std::int32_t>(0).data();
    for (const auto batch : batch_sizes) {
        const AllocationProbe probe;
        check(sum_where(columns, 0, predicates, batch) == expected, "SUM value, global row order and batch independence");
        check(sum_where(columns, 0, predicates, batch) == expected, "repeated execution");
        check(allocations::calls == 0, "successful SUM allocated");
        check(columns.column<std::int32_t>(0).data() == storage, "borrowed storage unchanged");
        ++checks;
    }
}

void known_sums() {
    const ColumnBatch columns({std::vector<std::int32_t>{-2, 0, 2, 2, 4, 6},
        std::vector<std::int64_t>{big, big + 1, big + 2, big + 3, big + 4, big + 5},
        std::vector<float>{-1, -0.0F, 0.0F, 0.5F, 1, 2}});
    successful(columns, {}, 12);
    const std::array<NumericConstant, 3> constants{std::int32_t{2}, big + 2, 0.0F};
    const std::array<std::array<std::int64_t, 6>, 3> expected{{{-2, 2, 10, 14, 4, 8},
        {-2, 0, 12, 14, 2, 10}, {-2, 0, 12, 14, 2, 10}}};
    for (std::size_t column = 0; column < 3; ++column) {
        for (std::size_t op = 0; op < comparisons.size(); ++op) {
            const std::array predicates{Predicate{column, comparisons[op], constants[column]}};
            successful(columns, predicates, expected[column][op]);
        }
    }
    const std::array partial{Predicate{0, Comparison::greater_equal, std::int32_t{2}},
                             Predicate{2, Comparison::less_equal, 1.0F}};
    successful(columns, partial, 8);
    const std::array three{partial[0], partial[1], Predicate{1, Comparison::not_equal, big + 3}};
    successful(columns, three, 6);
    const std::array none{Predicate{0, Comparison::less, std::int32_t{-10}}};
    successful(columns, none, 0);
    const ColumnBatch empty({std::vector<std::int32_t>{}, std::vector<std::int64_t>{}, std::vector<float>{}});
    successful(empty, {}, 0);
    successful(empty, partial, 0);
    const ColumnBatch single({std::vector<std::int32_t>{7}});
    successful(single, {}, 7);
    successful(single, std::array{Predicate{0, Comparison::equal, std::int32_t{7}}}, 7);
    successful(single, std::array{Predicate{0, Comparison::not_equal, std::int32_t{7}}}, 0);
    const ColumnBatch extremes({std::vector<std::int32_t>{high32, high32, low32, low32, high32, low32, 3}});
    successful(extremes, {}, 0);
    successful(extremes, std::array{Predicate{0, Comparison::greater, std::int32_t{0}}},
        3 * static_cast<std::int64_t>(high32) + 3);
    successful(extremes, std::array{Predicate{0, Comparison::less, std::int32_t{0}}},
        3 * static_cast<std::int64_t>(low32));

    const ColumnBatch other_target({std::vector<std::int64_t>{big, big + 1},
        std::vector<std::int32_t>{high32, -2}});
    const std::array different{Predicate{0, Comparison::greater_equal, big}};
    const auto expected_other = oracle(other_target, 1, different, 1);
    check(expected_other == static_cast<std::int64_t>(high32) - 2, "nonzero target index, other predicate column");
    for (const auto batch : batch_sizes) {
        const AllocationProbe probe;
        check(sum_where(other_target, 1, different, batch) == expected_other, "nonzero target column");
        check(allocations::calls == 0, "nonzero target allocated");
        ++checks;
    }
}

template <typename Error>
void rejected_by_both(const ColumnBatch& columns, std::size_t target_column,
                      std::span<const Predicate> predicates, std::size_t batch_size) {
    expect_throw<Error>([&] { (void)oracle(columns, target_column, predicates, batch_size); });
    expect_throw<Error>([&] { (void)sum_where(columns, target_column, predicates, batch_size); });
}

void rejection_and_order() {
    const ColumnBatch columns({std::vector<std::int32_t>{7}, std::vector<std::int64_t>{big}, std::vector<float>{0}});
    const ColumnBatch empty({std::vector<std::int32_t>{}, std::vector<std::int64_t>{}, std::vector<float>{}});
    const std::array out_of_range{Predicate{3, Comparison::equal, std::int32_t{0}}};
    const std::array malformed{Predicate{0, static_cast<Comparison>(-1), std::int32_t{0}}};
    const std::array wrong_first{Predicate{0, Comparison::equal, std::int64_t{7}}};
    for (const auto* input : {&columns, &empty}) {
        for (const auto batch : batch_sizes) {
            rejected_by_both<std::out_of_range>(*input, maximum, {}, batch);
            rejected_by_both<std::invalid_argument>(*input, 1, {}, batch);
            rejected_by_both<std::invalid_argument>(*input, 2, {}, batch);
            rejected_by_both<std::out_of_range>(*input, 0, out_of_range, batch);
            rejected_by_both<std::invalid_argument>(*input, 0, malformed, batch);
            rejected_by_both<std::out_of_range>(*input, maximum, malformed, batch); // Target before predicates.
            rejected_by_both<std::invalid_argument>(*input, 0, wrong_first, batch);
            for (const auto predicate : std::array{
                Predicate{1, Comparison::equal, 1.0F}, Predicate{2, Comparison::equal, std::int32_t{1}},
                Predicate{2, Comparison::equal, std::numeric_limits<float>::infinity()},
                Predicate{2, Comparison::not_equal, -std::numeric_limits<float>::infinity()},
                Predicate{2, Comparison::equal, std::numeric_limits<float>::quiet_NaN()}}) {
                const std::array bad{predicate};
                rejected_by_both<std::invalid_argument>(*input, 0, bad, batch);
            }
            const std::array none_then_bad{Predicate{0, Comparison::less, std::int32_t{-999}}, out_of_range[0]};
            rejected_by_both<std::out_of_range>(*input, 0, none_then_bad, batch); // Validate even after impossible filter.
            const std::array first_bad{wrong_first[0], out_of_range[0]};
            rejected_by_both<std::invalid_argument>(*input, 0, first_bad, batch); // Predicate order.
            const std::array index_first{out_of_range[0], wrong_first[0]};
            rejected_by_both<std::out_of_range>(*input, 0, index_first, batch);
        }
        rejected_by_both<std::invalid_argument>(*input, 0, {}, 0);
        rejected_by_both<std::invalid_argument>(*input, maximum, out_of_range, 0); // Batch before target.
    }
    for (const auto bad : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::quiet_NaN()}) {
        const ColumnBatch nonfinite({std::vector<std::int32_t>{7}, std::vector<float>{bad}});
        // Unused float data must be rejected, even if the filter selects nothing.
        rejected_by_both<std::invalid_argument>(nonfinite, 0, {}, 1);
        rejected_by_both<std::invalid_argument>(nonfinite, maximum, malformed, 1); // Data before target.
        const std::array none{Predicate{0, Comparison::less, std::int32_t{0}}};
        rejected_by_both<std::invalid_argument>(nonfinite, 0, none, 1);
        try { (void)sum_where(nonfinite, maximum, {}, 0); }
        catch (const std::invalid_argument& e) {
            check(std::string_view(e.what()) == "batch size must be positive", "batch before data validation");
            ++rejected;
            continue;
        }
        throw std::runtime_error("missing batch error");
    }
    const ColumnBatch no_columns({});
    rejected_by_both<std::out_of_range>(no_columns, 0, {}, 1);
    auto moved = ColumnBatch({std::vector<std::int32_t>{7}});
    const auto owner = std::move(moved);
    rejected_by_both<std::out_of_range>(moved, 0, {}, 1);
    successful(owner, {}, 7);
}

void checked_arithmetic_boundary() {
    // Negative overflow needs 2^32+1 INT32_MIN values; positive overflow needs
    // 2^32+3 INT32_MAX values. Each target alone exceeds 16 GiB. Do not allocate it.
    // These are direct arithmetic tests with near-limit accumulators, not claims
    // of an end-to-end overflowing dataset or of int32 values near int64 limits.
    constexpr auto positive_rows = static_cast<std::uint64_t>(high64 / high32) + 1;
    constexpr auto negative_rows = (std::uint64_t{1} << 32) + 1;
    static_assert(positive_rows == (std::uint64_t{1} << 32) + 3);
    static_assert(negative_rows * sizeof(std::int32_t) > (std::uint64_t{16} << 30));
    constexpr std::array<std::int64_t, 9> totals{low64, low64 + 1,
        low64 + static_cast<std::int64_t>(high32), -1, 0, 1,
        high64 - static_cast<std::int64_t>(high32), high64 - 1, high64};
    constexpr std::array<std::int32_t, 5> deltas{low32, -1, 0, 1, high32};
    for (const auto total : totals) {
        for (const auto delta : deltas) {
            bool overflow = false;
            std::int64_t expected = 0;
            try { expected = oracle_add(total, delta); }
            catch (const std::overflow_error&) { overflow = true; }
            if (overflow) {
                expect_throw<std::overflow_error>([&] { (void)detail::checked_sum_add(total, delta); });
            } else {
                check(detail::checked_sum_add(total, delta) == expected, "checked int32 addition at int64 boundary");
            }
            ++checks;
        }
    }
    check(detail::checked_sum_add(high64 - high32, high32) == high64, "exact positive limit");
    check(detail::checked_sum_add(low64 - static_cast<std::int64_t>(low32), low32) == low64, "exact negative limit");
}

void fixed_seed_differential() {
    std::mt19937_64 random(0x56463108BULL);
    for (std::size_t trial = 0; trial < 24; ++trial) {
        const auto rows = static_cast<std::size_t>(random() % 258);
        std::vector<std::int32_t> values(rows);
        std::vector<std::int64_t> wide(rows);
        std::vector<float> real(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            values[row] = static_cast<std::int32_t>(random() % 2001) - 1000;
            if (row % 5 == 0) { values[row] = row % 2 == 0 ? high32 : low32; }
            wide[row] = big + static_cast<std::int64_t>(random() % 9);
            real[row] = static_cast<float>(static_cast<int>(random() % 8) - 3) / 8.0F;
        }
        const ColumnBatch columns({std::move(values), std::move(wide), std::move(real)});
        successful(columns, {}, oracle(columns, 0, {}, 1));
        for (const auto comparison : comparisons) {
            const std::array predicates{Predicate{0, comparison, std::int32_t{0}},
                Predicate{1, comparison, big + 4}, Predicate{2, comparison, 0.0F}};
            // Check each predicate type alone as well as the combined AND.
            for (const auto& predicate : predicates) {
                const auto one = std::span(&predicate, 1);
                successful(columns, one, oracle(columns, 0, one, 1));
            }
            successful(columns, predicates, oracle(columns, 0, predicates, 1));
        }
    }
}

// Opt-in, small reproducible workload; not part of CTest and not a framework.
// Release command: vectorforge_sum_tests --workload. No files are generated.
void workload() {
    constexpr std::size_t rows = 65'536, samples = 9, repetitions = 11;
    std::vector<std::int32_t> values(rows), flags(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        values[row] = static_cast<std::int32_t>(row % 2001) - 1000;
        flags[row] = static_cast<std::int32_t>(row % 100);
    }
    const ColumnBatch columns({std::move(values), std::move(flags)});
    const std::array low{Predicate{1, Comparison::equal, std::int32_t{0}}};
    const std::array high{Predicate{1, Comparison::less, std::int32_t{90}}};
    const std::array<std::span<const Predicate>, 3> cases{std::span<const Predicate>{}, low, high};
    const std::array names{"all", "low_1pct", "high_90pct"};
    std::cout << "# rows=65536; values=row%2001-1000; flag=row%100; warmup=2; samples=9; repetitions=11; balanced order\n"
              << "selection,batch,implementation,median_us,min_us,max_us,allocations_per_query,checksum\n";
    using Clock = std::chrono::steady_clock;
    for (std::size_t scenario = 0; scenario < cases.size(); ++scenario) {
        const auto predicates = cases[scenario];
        const auto expected = oracle(columns, 0, predicates, 1);
        for (const auto batch : std::array<std::size_t, 2>{32, 1024}) {
            auto production = [&] { return sum_where(columns, 0, predicates, batch); };
            auto reference = [&] { return oracle(columns, 0, predicates, batch); };
            for (int warmup = 0; warmup < 2; ++warmup) {
                check(production() == expected && reference() == expected, "workload warmup correctness");
            }
            std::array<double, samples> actual{}, baseline{};
            auto measure = [&](auto function) {
                std::int64_t checksum = 0;
                double microseconds;
                {
                    const AllocationProbe probe;
                    const auto start = Clock::now();
                    for (std::size_t repeat = 0; repeat < repetitions; ++repeat) { checksum += function(); }
                    const auto end = Clock::now();
                    microseconds = std::chrono::duration<double, std::micro>(end - start).count() /
                                   static_cast<double>(repetitions);
                }
                check(allocations::calls == 0, "workload queries allocated");
                check(checksum == expected * static_cast<std::int64_t>(repetitions), "timed query checksum");
                return microseconds;
            };
            for (std::size_t sample = 0; sample < samples; ++sample) {
                if ((sample + scenario) % 2 == 0) {
                    actual[sample] = measure(production); baseline[sample] = measure(reference);
                } else {
                    baseline[sample] = measure(reference); actual[sample] = measure(production);
                }
            }
            auto report = [&](const char* implementation, auto timings) {
                std::sort(timings.begin(), timings.end());
                std::cout << names[scenario] << ',' << batch << ',' << implementation << ','
                          << timings[samples / 2] << ',' << timings.front() << ',' << timings.back()
                          << ",0," << expected << '\n';
            };
            report("production", actual); report("independent_oracle", baseline);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        check_allocation_probe();
        if (argc == 2 && std::string_view(argv[1]) == "--workload") {
            workload(); return 0;
        }
        check(argc == 1, "use no arguments or --workload");
        known_sums();
        rejection_and_order();
        checked_arithmetic_boundary();
        fixed_seed_differential();
        std::cout << "Filtered SUM: " << checks << " exact/batch/arithmetic checks, " << rejected
                  << " expected rejections; repeated queries allocate zero; overflow boundary tested without >16 GiB dataset\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
