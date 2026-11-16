#include "topk/topk.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <source_location>
#include <sstream>
#include <stdexcept>

namespace {

namespace bench = vectorforge::benchmark;
namespace topk = bench::topk;
using vectorforge::ScoredRecord;

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

void exact(std::span<const ScoredRecord> actual, std::span<const ScoredRecord> expected) {
    check(actual.size() == expected.size(), "result count");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        check(actual[i].id == expected[i].id, "ID and deterministic ordering");
        check(std::bit_cast<std::uint32_t>(actual[i].score) == std::bit_cast<std::uint32_t>(expected[i].score),
              "exact float32 score bits");
    }
}

void both(const std::vector<ScoredRecord>& input, std::size_t k, const std::vector<ScoredRecord>& expected) {
    const auto before = input;
    exact(topk::full_sort(input, k), expected);
    exact(topk::bounded_heap(input, k), expected);
    exact(input, before);
}

void hand_calculated_results() {
    const std::vector<ScoredRecord> ties{{9, 3}, {2, 3}, {4, 2}, {1, 3}};
    both(ties, 2, {{1, 3}, {2, 3}});
    both(ties, 1, {{1, 3}});
    both(ties, 0, {});
    both(ties, 99, {{1, 3}, {2, 3}, {9, 3}, {4, 2}});
    both(ties, std::numeric_limits<std::size_t>::max(), {{1, 3}, {2, 3}, {9, 3}, {4, 2}});
    both({{1, 5}, {2, 1}, {3, 4}, {4, 2}, {5, 6}}, 2, {{5, 6}, {1, 5}});
    both({}, 0, {});
    both({}, 1, {});
    both({{42, -5}}, std::numeric_limits<std::size_t>::max(), {{42, -5}});
    both({{9, -0.0F}, {2, 0.0F}, {4, -1}, {1, 0.0F}}, 3, {{1, 0.0F}, {2, 0.0F}, {9, -0.0F}});
    std::vector<ScoredRecord> ascending{{9, -2}, {7, 0}, {5, 1}, {3, 4}};
    both(ascending, 2, {{3, 4}, {5, 1}});
    std::reverse(ascending.begin(), ascending.end());
    both(ascending, 2, {{3, 4}, {5, 1}});
}

void configurations_and_validation() {
    const std::array<std::string_view, 12> args{"--candidates", "17", "--k", "3", "--distribution", "ties",
        "--warmup", "0", "--iterations", "3", "--seed", "42"};
    const auto options = topk::parse_options(args);
    check(options.custom && options.custom->candidates == 17 && options.custom->k == 3 &&
          options.custom->distribution == topk::Distribution::ties, "custom Top-K workload");
    check(options.warmup == 0 && options.iterations == 3 && options.seed == 42, "sampling controls");
    for (const auto bad : std::array{
        std::array<std::string_view, 2>{"--candidates", "-1"},
        std::array<std::string_view, 2>{"--candidates", "18446744073709551615"},
        std::array<std::string_view, 2>{"--k", "2x"},
        std::array<std::string_view, 2>{"--iterations", "2"},
        std::array<std::string_view, 2>{"--seed", "18446744073709551616"},
        std::array<std::string_view, 2>{"--distribution", "nan"},
        std::array<std::string_view, 2>{"--rows", "1"},
    }) { invalid([&] { (void)topk::parse_options(bad); }); }
    const std::array<std::string_view, 1> missing{"--k"};
    invalid([&] { (void)topk::parse_options(missing); });
    const std::array<std::string_view, 3> smoke_custom{"--smoke", "--candidates", "1"};
    invalid([&] { (void)topk::parse_options(smoke_custom); });
    topk::validate({"empty", 0, std::numeric_limits<std::size_t>::max(), topk::Distribution::random}, {});
    invalid([] { topk::validate({"bad", 1, 1, static_cast<topk::Distribution>(99)}, {}); });
    for (const auto bad : std::array{std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        const std::array input{ScoredRecord{1, bad}};
        invalid([&] { topk::validate_candidates(input); });
    }
    const std::array duplicate{ScoredRecord{1, 1}, ScoredRecord{1, 2}};
    invalid([&] { topk::validate_candidates(duplicate); });
    const auto defaults = topk::default_workloads();
    check(defaults.size() == 10, "small explicit matrix");
    for (const auto& workload : defaults) {
        topk::validate(workload, {});
        check(workload.candidates * sizeof(ScoredRecord) < 4 * 1024 * 1024, "laptop-sized inputs");
    }
}

void deterministic_distributions_and_differential() {
    constexpr std::array distributions{topk::Distribution::random, topk::Distribution::ties,
                                       topk::Distribution::ascending, topk::Distribution::descending};
    for (const auto distribution : distributions) {
        for (const auto count : std::array<std::size_t, 5>{0, 1, 2, 17, 127}) {
            for (const auto seed : std::array<std::uint64_t, 3>{0, 42, 1447440438}) {
                topk::Workload workload{"tiny", count, 3, distribution};
                const auto candidates = topk::make_candidates(workload, seed);
                check(candidates.size() == count, "expected candidate count");
                exact(candidates, topk::make_candidates(workload, seed));
                workload.k = 999;
                exact(candidates, topk::make_candidates(workload, seed));
                topk::validate_candidates(candidates); // Also verifies generated IDs are unique.
                for (std::size_t i = 1; i < count; ++i) {
                    if (distribution == topk::Distribution::ascending) {
                        check(candidates[i - 1].score <= candidates[i].score, "ascending scores");
                    } else if (distribution == topk::Distribution::descending) {
                        check(candidates[i - 1].score >= candidates[i].score, "descending scores");
                    }
                }
                if (distribution == topk::Distribution::ties) {
                    for (const auto& value : candidates) {
                        check(value.score >= -4 && value.score <= 3 && std::floor(value.score) == value.score,
                              "eight-level equal-score-heavy distribution");
                    }
                }
                for (const auto k : std::array<std::size_t, 6>{0, 1, 2, count, count + 1,
                                                               std::numeric_limits<std::size_t>::max()}) {
                    exact(topk::bounded_heap(candidates, k), topk::full_sort(candidates, k));
                }
            }
        }
    }
    const auto first = topk::make_candidates({"seed", 17, 2, topk::Distribution::random}, 42);
    const auto second = topk::make_candidates({"seed", 17, 2, topk::Distribution::random}, 43);
    check(first != second, "different seeds change the candidate sequence");
}

std::vector<std::vector<std::string>> read_csv(std::string_view text) {
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool in_quotes = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '"') {
            if (in_quotes && i + 1 < text.size() && text[i + 1] == '"') { field += '"'; ++i; }
            else { in_quotes = !in_quotes; }
        } else if (!in_quotes && (c == ',' || c == '\n')) {
            row.push_back(std::move(field)); field.clear();
            if (c == '\n') { rows.push_back(std::move(row)); row.clear(); }
        } else { field += c; }
    }
    check(!in_quotes && field.empty() && row.empty(), "complete valid CSV");
    return rows;
}

void timed_smoke_and_csv() {
    topk::Options options;
    options.warmup = 1;
    options.iterations = 3;
    options.smoke = true;
    const topk::Workload workload{"tiny, \"quoted\"", 17, 4, topk::Distribution::ties};
    const auto run = topk::run_workload(workload, options);
    check(run.measurements[0].checksum == run.measurements[1].checksum, "identical ordered results");
    auto info = bench::metadata();
    info.compiler = "compiler, \"quoted\"\nline";
    std::ostringstream output;
    topk::write_csv_header(output);
    topk::write_csv(output, workload, options, info, run);
    const auto rows = read_csv(output.str());
    check(rows.size() == 3 && rows[0].size() == 26 && rows[1].size() == 26 && rows[2].size() == 26, "Top-K CSV schema");
    check(rows[1][0] == workload.id && rows[1][16] == info.compiler, "quoted fields round-trip");
    check(rows[1][1] == "full_sort" && rows[2][1] == "bounded_heap", "algorithm labels");
    check(rows[1][2] == "17" && rows[1][3] == "4" && std::stod(rows[1][4]) == 4.0 / 17.0, "M, K and ratio");
    check(rows[1][5] == "ties" && rows[1][12] == "4" && rows[1][24] == "smoke", "distribution, count, mode");
    check(rows[1][25] == "immutable_candidates_to_ordered_topk_including_allocation_and_copy", "explicit timing boundary");
    for (const auto& result : run.measurements) {
        check(result.samples_ms.size() == 3 && result.timing.min_ms > 0 &&
              result.timing.min_ms <= result.timing.median_ms && result.timing.median_ms <= result.timing.max_ms,
              "multiple finite timing samples");
    }
    const topk::Workload empty{"empty", 0, 1, topk::Distribution::random};
    const auto empty_run = topk::run_workload(empty, options);
    std::ostringstream empty_csv;
    topk::write_csv_header(empty_csv);
    topk::write_csv(empty_csv, empty, options, info, empty_run);
    check(read_csv(empty_csv.str())[1][4].empty(), "K/M undefined for M=0 is an empty CSV field");
    check(topk::run_workload({"zero_k", 17, 0, topk::Distribution::random}, options).measurements[1].returned == 0,
          "timed K=0 smoke case");
}

} // namespace

int main() {
    try {
        hand_calculated_results();
        configurations_and_validation();
        deterministic_distributions_and_differential();
        timed_smoke_and_csv();
        std::cout << "PASS: hand-calculated Top-K, configuration, deterministic distributions, 360 exact differential cases, timing and CSV\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
