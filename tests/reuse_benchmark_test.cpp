#include "reuse.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <source_location>
#include <sstream>
#include <stdexcept>

namespace {

namespace bench = vectorforge::benchmark;
namespace reuse = bench::reuse;

void check(bool condition, const char* message,
           const std::source_location location = std::source_location::current()) {
    if (!condition) {
        throw std::runtime_error(std::string(location.file_name()) + ":" +
                                 std::to_string(location.line()) + ": " + message);
    }
}

template <typename Error, typename Function>
void expect_throw(Function&& function) {
    try { function(); }
    catch (const Error&) { return; }
    throw std::runtime_error("expected exception was not thrown");
}

void configuration() {
    const std::array<std::string_view, 18> args{"--rows", "11", "--dimension", "3", "--selectivity", "25",
        "--k", "2", "--batch-size", "4", "--reuse", "100", "--warmup", "0", "--iterations", "3", "--seed", "42"};
    const auto options = reuse::parse_options(args);
    check(options.sampling.custom && options.sampling.custom->rows == 11 && options.sampling.custom->dimension == 3,
          "existing query configuration is forwarded unchanged");
    check(options.custom_reuse == 100 && options.sampling.warmup == 0 && options.sampling.iterations == 3 &&
          options.sampling.seed == 42, "reuse and sampling configuration");
    for (const auto bad : std::array{
        std::array<std::string_view, 2>{"--reuse", "0"},
        std::array<std::string_view, 2>{"--reuse", "-1"},
        std::array<std::string_view, 2>{"--reuse", "1x"},
        std::array<std::string_view, 2>{"--reuse", "18446744073709551616"},
        std::array<std::string_view, 2>{"--dimension", "0"},
        std::array<std::string_view, 2>{"--iterations", "2"},
        std::array<std::string_view, 2>{"--unknown", "1"},
    }) { expect_throw<std::invalid_argument>([&] { (void)reuse::parse_options(bad); }); }
    const std::array<std::string_view, 1> missing{"--reuse"};
    expect_throw<std::invalid_argument>([&] { (void)reuse::parse_options(missing); });
    const std::array<std::string_view, 3> smoke_custom{"--smoke", "--reuse", "100"};
    expect_throw<std::invalid_argument>([&] { (void)reuse::parse_options(smoke_custom); });
    expect_throw<std::invalid_argument>([] { reuse::validate({{"bad"}, {}}, {}); });
    expect_throw<std::invalid_argument>([] { reuse::validate({{"bad"}, {0}}, {}); });
    const auto defaults = reuse::default_workloads();
    check(defaults.size() == 4, "four selected logical datasets, not a Cartesian matrix");
    for (const auto& workload : defaults) {
        reuse::validate(workload, {});
        check(workload.reuse_counts == std::vector<std::size_t>{1, 10, 100}, "representative modeled reuse counts");
        check(workload.query.rows * workload.query.dimension * sizeof(float) < 8 * 1024 * 1024,
              "small default embedding payload");
    }
}

void break_even_and_model() {
    const auto equal_at_five = reuse::model(10, 4, 2, 5);
    check(equal_at_five.legacy_total_ms == 20 && equal_at_five.trusted_total_ms == 20 &&
          equal_at_five.break_even_reuse == 6, "equality at R=5 requires R=6 for a strict win");
    const auto win_at_six = reuse::model(10, 4, 2, 6);
    check(win_at_six.legacy_total_ms == 24 && win_at_six.trusted_total_ms == 22 && win_at_six.break_even_reuse == 6,
          "hand-calculated admission plus six queries");
    check(reuse::model(9, 4, 2, 1).break_even_reuse == 5, "nonintegral break-even ratio");
    check(reuse::model(0, 1, 0.5, 1).break_even_reuse == 1, "zero admission and faster queries win at one reuse");
    check(!reuse::model(2, 1, 1, 10).break_even_reuse, "equal query costs never amortize admission");
    check(!reuse::model(2, 1, 1.5, 10).break_even_reuse, "slower trusted queries have no positive break-even");
    check(!reuse::model(0, 0, 0, 1).break_even_reuse, "equal zero costs are not a strict win");
    expect_throw<std::invalid_argument>([] { (void)reuse::model(1, 2, 1, 0); });
    expect_throw<std::invalid_argument>([] { (void)reuse::model(-1, 2, 1, 1); });
    expect_throw<std::invalid_argument>([] { (void)reuse::model(1, -2, 1, 1); });
    expect_throw<std::invalid_argument>([] { (void)reuse::model(1, 2, -1, 1); });
    expect_throw<std::invalid_argument>([] { (void)reuse::model(std::numeric_limits<double>::quiet_NaN(), 2, 1, 1); });
    expect_throw<std::invalid_argument>([] { (void)reuse::model(1, std::numeric_limits<double>::infinity(), 1, 1); });
    expect_throw<std::overflow_error>([] { (void)reuse::model(1, std::numeric_limits<double>::max(), 1, 2); });
    expect_throw<std::overflow_error>([] { (void)reuse::model(1, std::numeric_limits<double>::denorm_min(), 0, 1); });
}

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
            row.push_back(std::move(field)); field.clear();
            if (c == '\n') { rows.push_back(std::move(row)); row.clear(); }
        } else { field += c; }
    }
    check(!in_quotes && field.empty() && row.empty(), "complete CSV records");
    return rows;
}

void exact_reuse_smoke_and_csv() {
    reuse::Options options;
    options.sampling.warmup = 1;
    options.sampling.iterations = 3;
    options.sampling.smoke = true;
    const reuse::Workload workload{{"tiny, \"quoted\"", 17, 3, 35, 4, 5}, {1, 10, 100}};
    const auto run = reuse::run_workload(workload, options);
    check(run.qualifying == 5 && run.returned == 4, "trusted/legacy exact checks pass with expected selectivity and K");
    for (const auto* samples : {&run.admission, &run.legacy_query, &run.trusted_query}) {
        check(samples->values_ms.size() == 3 && samples->timing.min_ms > 0 &&
              samples->timing.min_ms <= samples->timing.median_ms && samples->timing.median_ms <= samples->timing.max_ms,
              "admission and query samples measured separately with positive timings");
    }
    auto info = bench::metadata();
    info.compiler = "compiler, \"quoted\"\nline";
    std::ostringstream output;
    reuse::write_csv_header(output);
    reuse::write_csv(output, workload, options, info, run);
    const auto rows = read_csv(output.str());
    check(rows.size() == 4 && rows[0].size() == 41, "one header and three modeled reuse rows");
    for (std::size_t i = 1; i < rows.size(); ++i) {
        const auto& row = rows[i];
        const auto count = workload.reuse_counts[i - 1];
        check(row.size() == 41 && row[0] == workload.query.id && row[32] == info.compiler, "valid quoted CSV schema");
        check(row[7] == std::to_string(count) && row[8] == "5" && row[9] == "4", "reuse and result counts");
        check(row[24] == "model_from_component_medians" && row[40] == "smoke", "explicit model/smoke labels");
        const auto expected = reuse::model(run.admission.timing.median_ms, run.legacy_query.timing.median_ms,
                                           run.trusted_query.timing.median_ms, count);
        check(std::stod(row[22]) == expected.legacy_total_ms && std::stod(row[23]) == expected.trusted_total_ms,
              "modeled costs include one admission and exactly R per-query medians");
        check(row[13] == rows[1][13] && row[16] == rows[1][16] && row[19] == rows[1][19],
              "reuse scenarios share measured components rather than pretending to be separate workflow measurements");
    }
    // Deterministic synthetic component samples exercise the no-break-even CSV case.
    auto synthetic = run;
    synthetic.admission = {{2, 2, 2}, {2, 2, 2}};
    synthetic.legacy_query = {{1, 1, 1}, {1, 1, 1}};
    synthetic.trusted_query = {{1, 1, 1}, {1, 1, 1}};
    std::ostringstream no_break_even;
    reuse::write_csv_header(no_break_even);
    reuse::write_csv(no_break_even, workload, options, info, synthetic);
    const auto no_rows = read_csv(no_break_even.str());
    check(no_rows[1][25].empty() && no_rows[1][26] == "trusted_not_faster", "explicit no-positive-break-even output");
}

void empty_small_and_reuse_identity() {
    reuse::Options options;
    options.sampling.warmup = 0;
    options.sampling.iterations = 3;
    options.sampling.smoke = true;
    const auto empty = reuse::run_workload({{"empty", 0, 3, 0, 10, 1}, {1, 10}}, options);
    check(empty.qualifying == 0 && empty.returned == 0, "empty source and snapshot are valid");
    const auto single = reuse::run_workload({{"single", 1, 2, 100, 10, 3}, {1}}, options);
    check(single.qualifying == 1 && single.returned == 1, "single-row query reuse");
    const auto zero_k = reuse::run_workload({{"zero_k", 3, 2, 100, 0, 2}, {100}}, options);
    check(zero_k.returned == 0, "K=0 remains fully validated through both APIs");
    const auto none = reuse::run_workload({{"none", 17, 3, 0, 4, 5}, {1}}, options);
    check(none.returned == 0 && none.qualifying == 0, "no-match query reuse");
    const auto first = reuse::run_workload({{"same", 11, 3, 25, 2, 4}, {1}}, options);
    const auto repeated = reuse::run_workload({{"same", 11, 3, 25, 2, 4}, {100}}, options);
    check(first.dataset_checksum == repeated.dataset_checksum && first.result_checksum == repeated.result_checksum,
          "R does not change logical data or queries");
    check(repeated.trusted_query.values_ms.size() == 3, "R=100 is a labeled model, not 100 timed or reconstructed admissions");
}

} // namespace

int main() {
    try {
        configuration();
        break_even_and_model();
        exact_reuse_smoke_and_csv();
        empty_small_and_reuse_identity();
        std::cout << "PASS: reuse configuration, strict break-even, exact workflows, admission/query samples and CSV\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
