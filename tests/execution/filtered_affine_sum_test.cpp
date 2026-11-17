#include "vectorforge/filtered_affine_sum.hpp"
#include "vectorforge/column_batch.hpp"
#include "vectorforge/filtered_count.hpp"
#include "vectorforge/filtered_sum.hpp"

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
#include <optional>
#include <random>
#include <stdexcept>
#include <string_view>
#include <utility>

// Single-threaded, ordinary new/new[] probe, as in the numeric COUNT/SUM tests.
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
constexpr auto low32 = std::numeric_limits<std::int32_t>::min();
constexpr auto high32 = std::numeric_limits<std::int32_t>::max();
constexpr auto low64 = std::numeric_limits<std::int64_t>::min();
constexpr auto high64 = std::numeric_limits<std::int64_t>::max();
constexpr auto power62 = std::int64_t{1} << 62;
constexpr std::int64_t big = 9'007'199'254'740'993LL;
constexpr std::array<std::size_t, 7> batches{1, 2, 7, 32, 33, 1024, maximum};
constexpr std::array operators{Comparison::less, Comparison::less_equal, Comparison::greater,
    Comparison::greater_equal, Comparison::equal, Comparison::not_equal};
std::size_t checks = 0, rejected = 0, compatibility = 0;
void check(bool condition, const char* message) { if (!condition) { throw std::runtime_error(message); } }
struct Probe {
    Probe() { allocations::calls = 0; allocations::active = true; }
    ~Probe() { allocations::active = false; }
};
void calibrate() {
    const Probe probe;
    void* p = ::operator new(8);
    const auto calls = allocations::calls;
    ::operator delete(p);
    check(calls == 1, "allocation probe is inactive");
}
template <typename Error, typename F> void throws(F function) {
    try { function(); }
    catch (const Error&) { ++rejected; return; }
    throw std::runtime_error("missing expected exception");
}

// Independent transform: unsigned magnitude multiplication and explicit sign,
// never negating INT32_MIN in int32. Product magnitude is at most 2^62.
std::int64_t oracle_value(std::int32_t x, std::int32_t a, std::int32_t b) {
    auto magnitude = [](std::int32_t value) {
        const auto wide = static_cast<std::int64_t>(value);
        return static_cast<std::uint64_t>(wide < 0 ? -wide : wide);
    };
    const auto magnitude_product = magnitude(x) * magnitude(a);
    const auto product = static_cast<std::int64_t>(magnitude_product);
    return ((x < 0) != (a < 0) ? -product : product) + static_cast<std::int64_t>(b);
}
std::int64_t oracle_add(std::int64_t total, std::int64_t value) {
    // Remaining room from total, independently of production's limit-minus-value.
    if ((total >= 0 && value > 0 && value > high64 - total) ||
        (total < 0 && value < 0 && value < low64 - total)) {
        throw std::overflow_error("oracle prefix overflow");
    }
    return total + value;
}

// Independent validation and row interpreter. No production validation, filter,
// transform or checked-add operation is called here. Private helper includes above
// are used ONLY by the separate validation-timing function at the end of this file.
std::int64_t oracle(const ColumnBatch& columns, std::size_t source_column,
                    std::int32_t a, std::int32_t b,
                    std::span<const Predicate> predicates, std::size_t batch_size) {
    if (batch_size == 0) { throw std::invalid_argument("oracle batch size"); }
    for (std::size_t index = 0; index < columns.column_count(); ++index) {
        const auto view = columns.column_view(index);
        std::visit([&](auto values) {
            if (values.size() != columns.row_count()) { throw std::invalid_argument("oracle shape"); }
        }, view);
        if (const auto* values = std::get_if<std::span<const float>>(&view)) {
            for (const auto value : *values) {
                if (!std::isfinite(value)) { throw std::invalid_argument("oracle float data"); }
            }
        }
    }
    const auto view = columns.column_view(source_column);
    const auto* source = std::get_if<std::span<const std::int32_t>>(&view);
    if (!source) { throw std::invalid_argument("oracle source type"); }
    for (const auto& predicate : predicates) {
        if (std::find(operators.begin(), operators.end(), predicate.comparison) == operators.end()) {
            throw std::invalid_argument("oracle operator");
        }
        if (columns.column_view(predicate.column_index).index() != predicate.constant.index()) {
            throw std::invalid_argument("oracle predicate type");
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
                throw std::invalid_argument("oracle operator");
            }, predicate.constant);
            if (!match) { selected = false; break; }
        }
        if (selected) { total = oracle_add(total, oracle_value((*source)[row], a, b)); }
    }
    return total;
}

void equivalent(const ColumnBatch& columns, std::size_t source, std::int32_t a, std::int32_t b,
                std::span<const Predicate> predicates, std::optional<std::int64_t> known = std::nullopt) {
    std::int64_t expected = 0;
    bool overflow = false;
    try { expected = oracle(columns, source, a, b, predicates, 1); }
    catch (const std::overflow_error&) { overflow = true; }
    if (known) { check(!overflow && expected == *known, "hand value disagrees with independent oracle"); }
    const auto* storage = columns.column<std::int32_t>(source).data();
    for (const auto batch : batches) {
        if (overflow) {
            // Failure may allocate its exception message; only success is probed.
            throws<std::overflow_error>([&] { (void)sum_affine_where(columns, source, a, b, predicates, batch); });
        } else {
            const Probe probe;
            check(sum_affine_where(columns, source, a, b, predicates, batch) == expected, "affine result/batch mismatch");
            check(sum_affine_where(columns, source, a, b, predicates, batch) == expected, "repeated result mismatch");
            check(allocations::calls == 0, "successful affine query allocated");
        }
        check(columns.column<std::int32_t>(source).data() == storage, "input storage changed");
        ++checks;
    }
}

void known_values_and_types() {
    const ColumnBatch columns({std::vector<std::int32_t>{-2, 0, 2, 2, 4, 6},
        std::vector<std::int64_t>{big, big + 1, big + 2, big + 3, big + 4, big + 5},
        std::vector<float>{-1, -0.0F, 0.0F, 0.5F, 1, 2}});
    equivalent(columns, 0, 1, 0, {}, 12);
    equivalent(columns, 0, 0, 0, {}, 0);
    equivalent(columns, 0, 0, 7, {}, 42);
    equivalent(columns, 0, -3, -5, {}, -66);
    equivalent(columns, 0, 7, -11, {}, 18);
    const std::array<NumericConstant, 3> constants{std::int32_t{2}, big + 2, 0.0F};
    for (std::size_t column = 0; column < constants.size(); ++column) {
        for (const auto op : operators) {
            const std::array predicates{Predicate{column, op, constants[column]}};
            equivalent(columns, 0, -3, -5, predicates);
            equivalent(columns, 0, 1, 0, predicates);
        }
    }
    const std::array partial{Predicate{0, Comparison::greater_equal, std::int32_t{2}},
        Predicate{2, Comparison::less_equal, 1.0F}, Predicate{1, Comparison::not_equal, big + 3}};
    equivalent(columns, 0, -3, -5, partial, -28);
    const std::array none{Predicate{0, Comparison::less, std::int32_t{-999}}};
    equivalent(columns, 0, low32, high32, none, 0);
    const ColumnBatch empty({std::vector<std::int32_t>{}, std::vector<std::int64_t>{}, std::vector<float>{}});
    equivalent(empty, 0, low32, high32, {}, 0); equivalent(empty, 0, 0, 0, partial, 0);
    const ColumnBatch other_source({std::vector<float>{0, 1}, std::vector<std::int32_t>{4, -1}});
    equivalent(other_source, 1, 2, 3, std::array{Predicate{0, Comparison::greater, 0.0F}}, 1);
    constexpr std::array<std::int32_t, 5> edge{low32, -1, 0, 1, high32};
    for (const auto x : edge) {
        const ColumnBatch one({std::vector<std::int32_t>{x}});
        for (const auto a : edge) { for (const auto b : edge) { equivalent(one, 0, a, b, {}); } }
    }
    const ColumnBatch min_source({std::vector<std::int32_t>{low32}});
    equivalent(min_source, 0, low32, high32, {}, power62 + high32);
    equivalent(min_source, 0, high32, low32, {}, -power62);
    auto moved = ColumnBatch({std::vector<std::int32_t>{7}});
    const auto owner = std::move(moved); equivalent(owner, 0, 1, 0, {}, 7);
    throws<std::out_of_range>([&] { (void)sum_affine_where(moved, 0, 0, 0, {}, 1); });
}

void public_overflow_and_order() {
    const ColumnBatch positive({std::vector<std::int32_t>{low32, low32, high32},
        std::vector<std::int32_t>{1, 0, 0}});
    // Two +2^62 values overflow; the third would cancel into range if regrouped.
    equivalent(positive, 0, low32, 0, {});
    for (const auto batch : batches) {
        throws<std::overflow_error>([&] { (void)sum_affine_where(positive, 0, low32, 0, {}, batch); });
    }
    const ColumnBatch first({std::vector<std::int32_t>{low32}});
    equivalent(first, 0, low32, 0, {}, power62);
    const ColumnBatch second({std::vector<std::int32_t>{low32, low32}});
    equivalent(second, 0, low32, 0, {}); // Small public-API two-row reproducer.
    const ColumnBatch reordered({std::vector<std::int32_t>{low32, high32, low32}});
    equivalent(reordered, 0, low32, 0, {}, power62 + (std::int64_t{1} << 31));
    // Filtering prevents an unselected row from entering the overflowing prefix.
    equivalent(positive, 0, low32, 0, std::array{Predicate{1, Comparison::equal, std::int32_t{1}}}, power62);
    const ColumnBatch exact_min({std::vector<std::int32_t>{low32, low32}});
    equivalent(exact_min, 0, high32, low32, {}, low64);
    const ColumnBatch negative({std::vector<std::int32_t>{low32, low32, low32, high32, high32}});
    equivalent(negative, 0, high32, low32, {}); // Later positive cancellation cannot rescue the third row.
    const std::array invalid{Predicate{maximum, Comparison::equal, std::int32_t{0}}};
    throws<std::out_of_range>([&] { (void)sum_affine_where(positive, 0, low32, 0, invalid, 1); });
    try { (void)sum_affine_where(second, 0, low32, 0, {}, 1); }
    catch (const std::overflow_error& e) {
        check(std::string_view(e.what()) == "filtered affine sum overflows int64", "overflow category/diagnostic");
        ++rejected; return;
    }
    throw std::runtime_error("missing first overflowing prefix");
}

template <typename Error>
void reject(const ColumnBatch& columns, std::size_t source, std::int32_t a, std::int32_t b,
            std::span<const Predicate> predicates, std::size_t batch) {
    throws<Error>([&] { (void)oracle(columns, source, a, b, predicates, batch); });
    throws<Error>([&] { (void)sum_affine_where(columns, source, a, b, predicates, batch); });
}
template <typename F> void diagnostic(F function, std::string_view expected) {
    try { function(); }
    catch (const std::invalid_argument& e) {
        check(e.what() == expected, "validation precedence mismatch"); ++rejected; return;
    }
    throw std::runtime_error("missing invalid_argument");
}
void invalid_inputs() {
    const ColumnBatch columns({std::vector<std::int32_t>{7}, std::vector<std::int64_t>{big}, std::vector<float>{0}});
    const ColumnBatch empty({std::vector<std::int32_t>{}, std::vector<std::int64_t>{}, std::vector<float>{}});
    const std::array bad_index{Predicate{3, Comparison::equal, std::int32_t{0}}};
    const std::array bad_type{Predicate{0, Comparison::equal, std::int64_t{7}}};
    const std::array malformed{Predicate{maximum, static_cast<Comparison>(-1), std::int32_t{0}}};
    for (const auto* input : {&columns, &empty}) {
        for (const auto coefficients : std::array<std::pair<std::int32_t, std::int32_t>, 3>{{{0, 0}, {0, 7}, {1, 0}}}) {
            const auto [a, b] = coefficients;
            reject<std::invalid_argument>(*input, maximum, a, b, bad_index, 0);
            for (const auto batch : batches) {
                reject<std::out_of_range>(*input, maximum, a, b, malformed, batch);
                reject<std::invalid_argument>(*input, 1, a, b, {}, batch);
                reject<std::invalid_argument>(*input, 2, a, b, {}, batch);
                reject<std::invalid_argument>(*input, 1, a, b, bad_index, batch); // Source type before predicate index.
                reject<std::out_of_range>(*input, 0, a, b, bad_index, batch);
                reject<std::invalid_argument>(*input, 0, a, b, malformed, batch);
                for (const auto bad : std::array{bad_type[0], Predicate{1, Comparison::less, 0.0F},
                    Predicate{2, Comparison::equal, std::int32_t{0}},
                    Predicate{2, Comparison::equal, std::numeric_limits<float>::quiet_NaN()},
                    Predicate{2, Comparison::less, std::numeric_limits<float>::infinity()},
                    Predicate{2, Comparison::not_equal, -std::numeric_limits<float>::infinity()}}) {
                    reject<std::invalid_argument>(*input, 0, a, b, std::array{bad}, batch);
                }
                const std::array none_bad{Predicate{0, Comparison::less, std::int32_t{-99}}, bad_index[0]};
                reject<std::out_of_range>(*input, 0, a, b, none_bad, batch);
                reject<std::invalid_argument>(*input, 0, a, b, std::array{bad_type[0], bad_index[0]}, batch);
                reject<std::out_of_range>(*input, 0, a, b, std::array{bad_index[0], bad_type[0]}, batch);
            }
        }
    }
    const ColumnBatch no_columns({}); reject<std::out_of_range>(no_columns, 0, 0, 0, {}, 1);
    for (const auto value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()}) {
        const ColumnBatch nonfinite({std::vector<std::int32_t>{7}, std::vector<float>{value}});
        reject<std::invalid_argument>(nonfinite, 0, 0, 0, {}, 1);
        reject<std::invalid_argument>(nonfinite, maximum, 0, 0, bad_index, 1);
        reject<std::invalid_argument>(nonfinite, 0, 0, 0,
            std::array{Predicate{0, Comparison::less, std::int32_t{0}}}, 1);
        diagnostic([&] { (void)sum_affine_where(nonfinite, maximum, 0, 0, bad_index, 0); },
                   "batch size must be positive");
    }
    // Legal retained vector-object aliases across outer vector moves; mutate only
    // before the call to exercise malformed column lengths, without layout tricks.
    std::vector<ColumnBatch::Column> storage{std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 0},
        std::vector<std::int32_t>{1, 2}};
    auto& alias = std::get<std::vector<std::int32_t>>(storage[1]);
    const ColumnBatch malformed_data(std::move(storage)); alias.pop_back();
    diagnostic([&] { (void)sum_affine_where(malformed_data, 1, 0, 0, {}, 1); }, "affine SUM float columns must be finite");
    diagnostic([&] { (void)oracle(malformed_data, 1, 0, 0, {}, 1); }, "oracle float data");
    std::vector<ColumnBatch::Column> shape_storage{std::vector<std::int32_t>{1, 2}, std::vector<float>{0, 1}};
    auto& shape_alias = std::get<std::vector<float>>(shape_storage[1]);
    const ColumnBatch bad_shape(std::move(shape_storage)); shape_alias.resize(1);
    reject<std::invalid_argument>(bad_shape, maximum, 0, 0, bad_index, maximum);
    diagnostic([&] { (void)sum_affine_where(bad_shape, maximum, 0, 0, bad_index, 1); },
               "numeric column length does not match row count");
}

void differential_and_existing_apis() {
    std::mt19937_64 random(0x56463110BULL);
    for (std::size_t trial = 0; trial < 16; ++trial) {
        const auto rows = trial == 0 ? std::size_t{0} : static_cast<std::size_t>(random() % 258);
        std::vector<std::int32_t> x(rows);
        std::vector<std::int64_t> wide(rows);
        std::vector<float> real(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            x[row] = static_cast<std::int32_t>(random() % 2001) - 1000;
            if (row % 7 == 0) { x[row] = row % 2 == 0 ? low32 : high32; }
            wide[row] = big + static_cast<std::int64_t>(random() % 9);
            real[row] = static_cast<float>(static_cast<int>(random() % 8) - 3) / 8.0F;
        }
        const ColumnBatch columns({std::move(x), std::move(wide), std::move(real)});
        auto compare = [&](std::span<const Predicate> predicates) {
            for (const auto coefficients : std::array<std::pair<std::int32_t, std::int32_t>, 4>{{{1, 0}, {0, -7},
                                                       {-3, -5}, {low32, high32}}}) {
                equivalent(columns, 0, coefficients.first, coefficients.second, predicates);
            }
            const auto expected = oracle(columns, 0, 1, 0, predicates, 1);
            for (const auto batch : batches) {
                const Probe probe;
                check(sum_where(columns, 0, predicates, batch) == expected, "identity/existing SUM changed");
                const auto count = count_where(columns, predicates, batch);
                check(sum_affine_where(columns, 0, 0, 7, predicates, batch) == static_cast<std::int64_t>(count) * 7,
                      "constant affine/raw COUNT mismatch");
                check(allocations::calls == 0, "existing/new successful queries allocated"); ++compatibility;
            }
        };
        compare({});
        for (const auto op : operators) {
            const std::array predicates{Predicate{0, op, std::int32_t{0}},
                Predicate{1, op, big + 4}, Predicate{2, op, 0.0F}};
            for (const auto& predicate : predicates) { compare(std::span(&predicate, 1)); }
            compare(predicates);
        }
    }
}

// Timing component only, separate from the independent oracle. It performs the
// same public pre-scan checks, without filtering/transform/accumulation. Separately
// measured component medians are not subtracted to claim pure execution latency.
std::int64_t validation_only(const ColumnBatch& columns, std::span<const Predicate> predicates) {
    for (std::size_t index = 0; index < columns.column_count(); ++index) {
        const auto view = columns.column_view(index);
        detail::validate_column_length(view, columns.row_count());
        detail::validate_column_floats(view, "affine SUM float columns must be finite");
    }
    (void)columns.column<std::int32_t>(0);
    detail::validate_predicates(columns, predicates);
    return static_cast<std::int64_t>(columns.row_count());
}

// Opt-in Release command: vectorforge_affine_sum_tests --workload. No materialized
// strategy or baseline speedup claim. Generation, references and reporting are
// outside timing; fixed coefficients/input bounds keep checksums well inside int64.
void workload() {
    constexpr std::uint64_t seed = 0x56463110BULL;
    constexpr std::int32_t a = -7, b = 13;
    constexpr std::size_t samples = 9, repetitions = 5;
    using Clock = std::chrono::steady_clock;
    std::cout << "# seed=" << seed << "; x in [-1000,1000]; flags=rng%100; unused_float=(row%17)/8; warmup=2; samples=9; repetitions=5; alternating order\n"
              << "rows,batch,selectivity_percent,selected,a,b,scope,median_us,min_us,max_us,allocations_per_query,result\n";
    for (const auto rows : std::array<std::size_t, 2>{65'536, 1'000'000}) {
        std::mt19937_64 random(seed);
        std::vector<std::int32_t> x(rows), flags(rows);
        std::vector<float> floats(rows);
        std::array<std::size_t, 2> selected{};
        for (std::size_t row = 0; row < rows; ++row) {
            x[row] = static_cast<std::int32_t>(random() % 2001) - 1000;
            flags[row] = static_cast<std::int32_t>(random() % 100);
            floats[row] = static_cast<float>(row % 17) / 8.0F;
            if (flags[row] < 1) { ++selected[0]; }
            if (flags[row] < 90) { ++selected[1]; }
        }
        const ColumnBatch columns({std::move(x), std::move(flags), std::move(floats)});
        for (const auto batch : std::array<std::size_t, 2>{32, 1024}) {
            for (std::size_t scenario = 0; scenario < 2; ++scenario) {
                const auto percent = scenario == 0 ? std::int32_t{1} : std::int32_t{90};
                const std::array predicates{Predicate{1, Comparison::less, percent}};
                const auto expected = oracle(columns, 0, a, b, predicates, 1);
                auto query = [&] { return sum_affine_where(columns, 0, a, b, predicates, batch); };
                auto validation = [&] { return validation_only(columns, predicates); };
                for (int warmup = 0; warmup < 2; ++warmup) {
                    check(query() == expected && validation() == static_cast<std::int64_t>(rows), "workload warmup");
                }
                auto measure = [&](auto function, std::int64_t expected_value) {
                    std::int64_t checksum = 0;
                    double us;
                    {
                        const Probe probe;
                        const auto start = Clock::now();
                        for (std::size_t repeat = 0; repeat < repetitions; ++repeat) { checksum += function(); }
                        const auto finish = Clock::now();
                        us = std::chrono::duration<double, std::micro>(finish - start).count() / static_cast<double>(repetitions);
                    }
                    check(allocations::calls == 0, "workload query allocated");
                    check(checksum == expected_value * static_cast<std::int64_t>(repetitions), "workload checksum");
                    return us;
                };
                std::array<double, samples> total{}, validation_times{};
                for (std::size_t sample = 0; sample < samples; ++sample) {
                    if ((sample + scenario) % 2 == 0) {
                        total[sample] = measure(query, expected); validation_times[sample] = measure(validation, static_cast<std::int64_t>(rows));
                    } else {
                        validation_times[sample] = measure(validation, static_cast<std::int64_t>(rows)); total[sample] = measure(query, expected);
                    }
                }
                auto report = [&](const char* scope, auto times, std::int64_t result) {
                    std::sort(times.begin(), times.end());
                    std::cout << rows << ',' << batch << ',' << percent << ',' << selected[scenario] << ',' << a << ',' << b
                              << ',' << scope << ',' << times[samples / 2] << ',' << times.front() << ',' << times.back()
                              << ",0," << result << '\n';
                };
                report("public_total", total, expected);
                report("validation_only", validation_times, static_cast<std::int64_t>(rows));
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
        known_values_and_types(); public_overflow_and_order(); invalid_inputs(); differential_and_existing_apis();
        std::cout << "Affine SUM: " << checks << " exact batch/differential checks; " << rejected << " expected rejections; "
                  << compatibility << " existing SUM/COUNT comparisons; zero successful query allocations\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
