#include "benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <source_location>
#include <sstream>
#include <stdexcept>

namespace {

namespace bench = vectorforge::benchmark;

void check(bool condition, const char* message,
           const std::source_location location = std::source_location::current()) {
    if (!condition) {
        throw std::runtime_error(std::string(location.file_name()) + ":" +
                                 std::to_string(location.line()) + ": " + message);
    }
}

template <typename Function>
void invalid(Function&& function) {
    try { function(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected invalid_argument");
}

void configuration() {
    const std::array<std::string_view, 16> args{"--rows", "11", "--dimension", "3", "--selectivity", "25",
        "--k", "2", "--batch-size", "4", "--warmup", "0", "--iterations", "3", "--seed", "42"};
    const auto options = bench::parse_options(args);
    check(options.custom && options.custom->rows == 11 && options.custom->dimension == 3 &&
          options.custom->selectivity_percent == 25 && options.custom->k == 2 && options.custom->batch_size == 4,
          "custom workload options");
    check(options.warmup == 0 && options.iterations == 3 && options.seed == 42, "sampling options");
    for (const auto args_bad : std::array{
        std::array<std::string_view, 2>{"--dimension", "0"},
        std::array<std::string_view, 2>{"--batch-size", "0"},
        std::array<std::string_view, 2>{"--selectivity", "101"},
        std::array<std::string_view, 2>{"--iterations", "2"},
        std::array<std::string_view, 2>{"--rows", "-1"},
        std::array<std::string_view, 2>{"--rows", "1x"},
        std::array<std::string_view, 2>{"--seed", "18446744073709551616"},
        std::array<std::string_view, 2>{"--unknown", "1"},
    }) { invalid([&] { (void)bench::parse_options(args_bad); }); }
    const std::array<std::string_view, 1> missing{"--rows"};
    invalid([&] { (void)bench::parse_options(missing); });
    const std::array<std::string_view, 3> smoke_custom{"--smoke", "--rows", "1"};
    invalid([&] { (void)bench::parse_options(smoke_custom); });
    invalid([] { bench::validate({"overflow", std::numeric_limits<std::size_t>::max(), 2, 100, 1, 1}, {}); });
    invalid([] { bench::validate({"oversized_query", 0, std::numeric_limits<std::size_t>::max(), 0, 0, 1}, {}); });
    bench::validate({"empty", 0, 3, 0, 0, 1}, {});
    bench::validate({"huge_k", 1, 3, 100, std::numeric_limits<std::size_t>::max(), 1}, {});
    const auto matrix = bench::default_workloads();
    check(matrix.size() == 8, "small explicit default matrix");
    for (const auto& workload : matrix) {
        bench::validate(workload, {});
        check(workload.rows * workload.dimension * sizeof(float) < 32 * 1024 * 1024,
              "default embeddings stay below 32 MiB");
    }
}

void deterministic_data_and_selectivity() {
    bench::Workload workload{"tiny", 11, 3, 25, 2, 4};
    const auto first = bench::make_dataset(workload, 42);
    const auto second = bench::make_dataset(workload, 42);
    check(first.checksum == second.checksum && first.ids == second.ids && first.query == second.query,
          "fixed-seed reproducibility");
    check(first.columns.row_count() == 11 && first.embeddings.row_count() == 11 && first.embeddings.dimension() == 3,
          "workload shape");
    check(first.qualifying == 2, "floor(11 * 25 / 100) is exactly two candidates");
    const auto flags = first.columns.column<std::int32_t>(0);
    check(std::count(flags.begin(), flags.end(), 1) == 2, "exact selectivity, not Bernoulli approximation");
    for (std::size_t row = 0; row < 11; ++row) {
        check(first.ids[row] == 11 - row, "IDs are stable and differ from row indices");
        check(std::ranges::equal(first.embeddings.row(row), second.embeddings.row(row)), "same embedding payload");
        for (const auto value : first.embeddings.row(row)) {
            check(std::isfinite(value) && value >= -1 && value <= 1, "finite bounded embeddings");
        }
    }
    workload.k = 999;
    workload.batch_size = 1;
    check(bench::make_dataset(workload, 42).checksum == first.checksum, "K and B do not change logical data");
    check(bench::make_dataset(workload, 43).checksum != first.checksum, "seed changes data");
    workload.selectivity_percent = 100;
    const auto all = bench::make_dataset(workload, 42);
    check(all.qualifying == 11 && all.query == first.query, "full selectivity and stable query");
    check(std::ranges::equal(all.embeddings.row(0), first.embeddings.row(0)), "selectivity does not change embeddings");
    workload.selectivity_percent = 0;
    check(bench::make_dataset(workload, 42).qualifying == 0, "zero selectivity");
}

void statistics_and_build_guard() {
    const auto odd = bench::summarize(std::array{9.0, 1.0, 5.0});
    const auto even = bench::summarize(std::array{8.0, 4.0, 2.0, 6.0});
    check(odd.median_ms == 5 && odd.min_ms == 1 && odd.max_ms == 9, "odd median and min/max");
    check(even.median_ms == 5 && even.min_ms == 2 && even.max_ms == 8, "even median");
    invalid([] { (void)bench::summarize({}); });
    invalid([] { (void)bench::summarize(std::array{0.0, 1.0, 2.0}); });
    invalid([] { (void)bench::summarize(std::array{std::numeric_limits<double>::quiet_NaN()}); });
    auto info = bench::metadata();
    info.build_type = "Debug";
    info.sanitized = false;
    invalid([&] { bench::require_performance_build(info, false); });
    bench::require_performance_build(info, true);
    info.build_type = "Release";
    info.sanitized = true;
    invalid([&] { bench::require_performance_build(info, false); });
    info.sanitized = false;
    bench::require_performance_build(info, false);
}

// Minimal CSV reader for the writer's output, including quoted commas, quotes
// and newlines. Keeping this independent catches malformed machine-readable rows.
std::vector<std::vector<std::string>> read_csv(std::string_view text) {
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool in_quotes = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const auto c = text[i];
        if (c == '"') {
            if (in_quotes && i + 1 < text.size() && text[i + 1] == '"') { field += '"'; ++i; }
            else { in_quotes = !in_quotes; }
        } else if (!in_quotes && (c == ',' || c == '\n')) {
            row.push_back(std::move(field));
            field.clear();
            if (c == '\n') { rows.push_back(std::move(row)); row.clear(); }
        } else { field += c; }
    }
    check(!in_quotes && field.empty() && row.empty(), "complete quoted CSV records");
    return rows;
}

void equivalence_and_csv() {
    const bench::Workload workload{"tiny, \"quoted\"", 17, 3, 35, 4, 5};
    bench::Options options;
    options.warmup = 1;
    options.iterations = 3;
    options.smoke = true;
    const auto run = bench::run_workload(workload, options);
    check(run.qualifying == 5, "expected qualifying count");
    check(run.measurements[0].returned == 4 && run.measurements[1].returned == 4 &&
          run.measurements[0].checksum == run.measurements[1].checksum, "equivalent query outputs");
    auto info = bench::metadata();
    info.compiler = "compiler, \"quoted\"\nnext line";
    std::ostringstream output;
    bench::write_csv_header(output);
    bench::write_csv(output, workload, options, info, run);
    const auto csv = read_csv(output.str());
    check(csv.size() == 3 && csv[0].size() == 29 && csv[1].size() == 29 && csv[2].size() == 29,
          "CSV header and two executor rows have the same schema");
    check(csv[1][0] == workload.id && csv[1][20] == info.compiler, "CSV quotes and embedded newlines round-trip");
    check(csv[1][1] == "reference_full_sort" && csv[2][1] == "batched_bounded_heap", "clear executor names");
    check(csv[1][5] == "0" && csv[2][5] == "5", "batch size is N/A=0 for the reference");
    check(csv[1][8] == "5" && csv[1][9] == "4" && csv[1][28] == "smoke", "candidate/result/mode metadata");
    check(std::stod(csv[1][7]) == 5.0 / 17.0, "actual selectivity reports the rounded candidate fraction");
    for (std::size_t executor = 0; executor < 2; ++executor) {
        const auto& measurement = run.measurements[executor];
        check(measurement.samples_ms.size() == 3 && measurement.timing.min_ms > 0 &&
              measurement.timing.min_ms <= measurement.timing.median_ms &&
              measurement.timing.median_ms <= measurement.timing.max_ms, "multiple valid timing measurements");
        check(std::stod(csv[executor + 1][16]) > 0, "numeric rows-per-second metric");
    }
    auto edge = workload;
    edge.rows = 0;
    check(bench::run_workload(edge, options).measurements[0].returned == 0, "empty workload");
    edge.rows = 1;
    edge.selectivity_percent = 100;
    edge.k = 0;
    check(bench::run_workload(edge, options).measurements[1].returned == 0, "K=0 workload");
}

} // namespace

int main() {
    try {
        configuration();
        deterministic_data_and_selectivity();
        statistics_and_build_guard();
        equivalence_and_csv();
        std::cout << "PASS: benchmark configuration, deterministic data/selectivity, statistics/build guards, exact equivalence and CSV\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
