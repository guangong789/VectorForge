#include "similarity/similarity.hpp"

#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
namespace bench = vectorforge::benchmark;
namespace sim = bench::similarity;

void check(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <typename Error, typename Function>
void expect_throw(Function&& function) {
    try { function(); }
    catch (const Error&) { return; }
    throw std::runtime_error("expected exception was not thrown");
}

void configuration_and_data() {
    const std::array<std::string_view, 10> args{"--rows", "19", "--dimension", "7",
        "--warmup", "0", "--iterations", "3", "--seed", "42"};
    const auto options = sim::parse_options(args);
    check(options.custom && options.custom->rows == 19 && options.custom->dimension == 7 &&
          options.sampling.warmup == 0 && options.sampling.iterations == 3 && options.sampling.seed == 42,
          "custom shape and common sampling options");
    const std::array<std::string_view, 2> dimension{"--dimension", "5"};
    check(sim::parse_options(dimension).custom->rows == 1024, "custom dimension uses small default N");
    for (const auto bad : std::array{
        std::array<std::string_view, 2>{"--dimension", "0"},
        std::array<std::string_view, 2>{"--rows", "-1"},
        std::array<std::string_view, 2>{"--rows", "1x"},
        std::array<std::string_view, 2>{"--rows", "18446744073709551616"},
        std::array<std::string_view, 2>{"--iterations", "2"},
        std::array<std::string_view, 2>{"--k", "10"},
        std::array<std::string_view, 2>{"--selectivity", "100"},
        std::array<std::string_view, 2>{"--batch-size", "4"},
    }) { expect_throw<std::invalid_argument>([&] { (void)sim::parse_options(bad); }); }
    const std::array<std::string_view, 1> missing{"--rows"};
    expect_throw<std::invalid_argument>([&] { (void)sim::parse_options(missing); });
    const std::array<std::string_view, 3> mixed{"--smoke", "--dimension", "5"};
    expect_throw<std::invalid_argument>([&] { (void)sim::parse_options(mixed); });
    expect_throw<std::invalid_argument>([] {
        sim::validate({"overflow", std::numeric_limits<std::size_t>::max(), 2}, {});
    });
    expect_throw<std::invalid_argument>([] {
        sim::validate({"huge_query", 0, std::numeric_limits<std::size_t>::max()}, {});
    });
    const auto defaults = sim::default_workloads();
    check(defaults.size() == 6, "small six-case matrix");
    for (const auto& workload : defaults) {
        sim::validate(workload, {});
        check(workload.rows == 1024 || workload.rows == 8192, "two candidate counts");
        check(workload.dimension == 32 || workload.dimension == 128 || workload.dimension == 768,
              "representative dimensions");
        check(workload.rows * workload.dimension * sizeof(float) <= 24 * 1024 * 1024,
              "default embedding payload at most 24 MiB");
    }
    const sim::Workload workload{"repeatable", 19, 7};
    const auto first = sim::make_data(workload, 42);
    const auto same = sim::make_data(workload, 42);
    const auto other = sim::make_data(workload, 43);
    check(first.query == same.query && first.embeddings == same.embeddings && first.checksum == same.checksum,
          "fixed seed reproduces all data");
    check(first.checksum != other.checksum && first.query != other.query, "seed changes data");
    check(first.query.size() == 7 && first.embeddings.size() == 133, "contiguous row-major shape");
    for (const auto value : first.embeddings) {
        check(std::isfinite(value) && std::abs(value) <= 1, "bounded finite generated values");
    }
}

void kernel_contract() {
    const std::array query{1.0F, 2.0F, 3.0F};
    const std::array embeddings{4.0F, 5.0F, 6.0F, -1.0F, 0.0F, 1.0F};
    std::array<float, 2> scores{};
    sim::sequential_fma_scores(query, embeddings, scores);
    check(scores == std::array{32.0F, 2.0F}, "known row-major scores");
    const float a = std::ldexp(1.0F, -23);
    const std::array fused_query{-1.0F, 1.0F + a};
    const std::array fused_embeddings{1.0F, 1.0F - a};
    std::array<float, 1> one{};
    sim::sequential_fma_scores(fused_query, fused_embeddings, one);
    check(one[0] == -std::ldexp(1.0F, -46), "sequential FMA retains fused residual");
    const std::array negative_zero{-0.0F};
    const std::array positive_one{1.0F};
    sim::sequential_fma_scores(negative_zero, positive_one, one);
    check(std::bit_cast<std::uint32_t>(one[0]) == 0, "accumulation starts at positive zero");
    sim::sequential_fma_scores(query, {}, {});
    expect_throw<std::invalid_argument>([&] { sim::sequential_fma_scores({}, {}, {}); });
    expect_throw<std::invalid_argument>([&] { sim::sequential_fma_scores(query, embeddings, one); });
    expect_throw<std::invalid_argument>([&] { sim::sequential_fma_scores(query, positive_one, one); });
    const std::array overflow_query{2.0F, -2.0F};
    const std::array overflow_embeddings{std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    expect_throw<std::overflow_error>([&] { sim::sequential_fma_scores(overflow_query, overflow_embeddings, one); });
    check(!sim::kernel_isa().empty(), "kernel compile target reported");
}

std::vector<std::vector<std::string>> read_csv(std::string_view text) {
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const auto c = text[i];
        if (c == '"') {
            if (quoted && i + 1 < text.size() && text[i + 1] == '"') { field += '"'; ++i; }
            else { quoted = !quoted; }
        } else if (!quoted && (c == ',' || c == '\n')) {
            row.push_back(std::move(field)); field.clear();
            if (c == '\n') { rows.push_back(std::move(row)); row.clear(); }
        } else { field += c; }
    }
    check(!quoted && field.empty() && row.empty(), "complete CSV");
    return rows;
}

void small_runs_and_csv() {
    sim::Options options;
    options.sampling.smoke = true;
    options.sampling.warmup = 1;
    options.sampling.iterations = 3;
    const sim::Workload workload{"tiny, \"quoted\"", 17, 5};
    const auto run = sim::run_workload(workload, options);
    const auto repeat = sim::run_workload(workload, options);
    check(run.dataset_checksum == repeat.dataset_checksum && run.result_checksum == repeat.result_checksum,
          "production reference and every kernel result checked exactly, repeatable fingerprints");
    check(run.samples_ms.size() == 3 && run.timing.min_ms > 0 &&
          run.timing.min_ms <= run.timing.median_ms && run.timing.median_ms <= run.timing.max_ms,
          "positive samples and variability");
    const bench::Metadata info{"GCC, \"test\"", "Debug", "-g", "cpu", "os", "line1\nline2", "UTC", true};
    std::ostringstream output;
    sim::write_csv_header(output);
    sim::write_csv(output, workload, options, info, run);
    const auto csv = read_csv(output.str());
    check(csv.size() == 2 && csv[0].size() == 27 && csv[1].size() == 27, "27-column CSV");
    check(csv[1][0] == workload.id && csv[1][18] == info.compiler && csv[1][23] == info.environment,
          "CSV escapes commas, quotes and multiline metadata");
    check(csv[1][4] == "340" && csv[1][25] == "1" && csv[1][26] == "smoke", "bytes and mode metadata");
    check(std::stod(csv[1][11]) == run.timing.median_ms * 1e6 / 17, "per-dot latency derived from scan");
    for (const auto& shape : std::array<sim::Workload, 5>{{{"empty", 0, 3}, {"one", 1, 1},
                                       {"d32", 7, 32}, {"d128", 3, 128}, {"d768", 2, 768}}}) {
        const auto small = sim::run_workload(shape, options);
        check(small.samples_ms.size() == 3, "tiny boundary runs pass exact production comparison");
        if (shape.rows == 0) {
            std::ostringstream empty;
            sim::write_csv_header(empty);
            sim::write_csv(empty, shape, options, info, small);
            check(read_csv(empty.str())[1][11].empty(), "empty input has no per-dot latency");
        }
    }
}

} // namespace

int main() {
    try {
        configuration_and_data();
        kernel_contract();
        small_runs_and_csv();
        std::cout << "PASS similarity configuration, sequential FMA, reference equivalence, repeatability and CSV\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
