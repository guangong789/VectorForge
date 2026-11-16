#include "similarity/accumulator.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
namespace bench = vectorforge::benchmark;
namespace sim = bench::similarity;
namespace study = bench::accumulator;

void check(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <typename Error, typename Function>
void expect_throw(Function&& function) {
    try { function(); }
    catch (const Error&) { return; }
    throw std::runtime_error("expected exception was not thrown");
}

// Independent indexing/control flow: lane chosen per feature, not chunked loads.
float oracle(std::span<const float> query, std::span<const float> values) {
    std::array<float, 4> lanes{};
    for (std::size_t j = 0; j < query.size(); ++j) {
        lanes[j % 4] = std::fma(query[j], values[j], lanes[j % 4]);
    }
    float result = lanes[0] + lanes[1];
    result = result + lanes[2];
    return result + lanes[3];
}

void tails_and_boundaries() {
    std::vector<std::size_t> dimensions;
    for (std::size_t d = 1; d <= 20; ++d) { dimensions.push_back(d); }
    for (const auto d : std::array<std::size_t, 9>{31, 32, 33, 127, 128, 129, 767, 768, 769}) {
        dimensions.push_back(d);
    }
    for (const auto d : dimensions) {
        for (const auto seed : std::array<std::uint64_t, 2>{0, 42}) {
            const auto data = sim::make_data({"tails", 7, d}, seed);
            std::vector<float> scores(7), sequential(7), original(7);
            sim::hardware_target::sequential_fma_scores(data.query, data.embeddings, sequential);
            sim::sequential_fma_scores(data.query, data.embeddings, original);
            check(study::differences(original, sequential).equal_bits == 7, "sequential baseline unchanged");
            for (int repetition = 0; repetition < 3; ++repetition) {
                std::fill(scores.begin(), scores.end(), std::numeric_limits<float>::quiet_NaN());
                study::fourway_scores(data.query, data.embeddings, scores);
                for (std::size_t row = 0; row < scores.size(); ++row) {
                    const auto expected = oracle(data.query, std::span(data.embeddings).subspan(row * d, d));
                    check(std::bit_cast<std::uint32_t>(scores[row]) == std::bit_cast<std::uint32_t>(expected),
                          "every tail/row matches independent fixed-order oracle repeatedly");
                }
            }
        }
    }
    const std::array one{1.0F};
    std::array<float, 1> score{};
    study::fourway_scores(one, {}, {});
    expect_throw<std::invalid_argument>([&] { study::fourway_scores({}, {}, {}); });
    expect_throw<std::invalid_argument>([&] { study::fourway_scores(one, {}, score); });
    const std::array three{1.0F, 1.0F, 1.0F};
    expect_throw<std::invalid_argument>([&] { study::fourway_scores(three, one, score); });
    const std::array denorm{std::numeric_limits<float>::denorm_min()};
    study::fourway_scores(one, denorm, score);
    check(std::bit_cast<std::uint32_t>(score[0]) == 1, "subnormal retained");
    const float a = std::ldexp(1.0F, -23);
    const std::array query{-1.0F, 1.0F + a}, values{1.0F, 1.0F - a};
    sim::hardware_target::sequential_fma_scores(query, values, score);
    check(score[0] == -std::ldexp(1.0F, -46), "sequential fused residual");
    study::fourway_scores(query, values, score);
    check(std::bit_cast<std::uint32_t>(score[0]) == 0, "changed order legitimately loses cancellation residual");

    const float largest = std::numeric_limits<float>::max();
    const std::array<float, 5> q5{1, 1, 1, 1, 1}, values5{largest, -largest, 0, 0, largest};
    sim::hardware_target::sequential_fma_scores(q5, values5, score);
    check(score[0] == largest, "sequential stays finite");
    expect_throw<std::overflow_error>([&] { study::fourway_scores(q5, values5, score); });
    const std::array<float, 8> q8{1, 1, 1, 1, 1, 1, 1, 1};
    const std::array values8{largest, largest, -largest, -largest, -largest, -largest, largest, largest};
    expect_throw<std::overflow_error>([&] { sim::hardware_target::sequential_fma_scores(q8, values8, score); });
    study::fourway_scores(q8, values8, score);
    check(score[0] == 0, "lane-wise cancellation can remain finite when sequential overflows");
    const std::array<float, 4> q4{1, 1, 1, 1}, merge_overflow{largest, largest, 0, 0};
    expect_throw<std::overflow_error>([&] { study::fourway_scores(q4, merge_overflow, score); });
    const std::array bad{std::numeric_limits<float>::quiet_NaN()};
    expect_throw<std::overflow_error>([&] { study::fourway_scores(one, bad, score); });
}

void numerical_reporting() {
    const float next = std::nextafter(1.0F, 2.0F), denorm = std::numeric_limits<float>::denorm_min();
    const std::array reference{1.0F, -1.0F, 0.0F, -0.0F};
    const std::array changed{next, -next, denorm, 0.0F};
    const auto d = study::differences(reference, changed);
    check(d.count == 4 && d.equal_bits == 0 && d.near_zero_reference == 2 && d.max_ulp == 1,
          "bit equality, near-zero counts, signed-zero and one-ULP distances");
    check(d.max_absolute == std::ldexp(1.0, -23) && d.max_relative == std::ldexp(1.0, -23),
          "hand-calculated absolute and floored relative error");
    const auto zero = study::differences(std::array{0.0F}, std::array{1.0F});
    check(zero.max_relative == 1 / study::relative_floor && zero.max_absolute == 1,
          "zero reference never divides by zero");
    check(study::differences(std::array{-denorm}, std::array{denorm}).max_ulp == 2,
          "monotone ULP distance crosses zero without double-counting signed zeros");
    check(study::differences(reference, reference).equal_bits == 4, "identical bits counted");
    check(study::differences({}, {}).count == 0, "empty metrics defined without fake equality rate");
    expect_throw<std::invalid_argument>([&] { (void)study::differences(reference, {}); });
    for (const auto floor : std::array{0.0, -1.0, std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::quiet_NaN()}) {
        expect_throw<std::invalid_argument>([&] { (void)study::differences({}, {}, floor); });
    }
    expect_throw<std::invalid_argument>([&] {
        (void)study::differences(std::array{std::numeric_limits<float>::infinity()}, std::array{0.0F});
    });
}

void ranking_sensitivity() {
    const auto cases = study::ranking_study();
    check(cases.size() == 4, "close scores, exact-tie control and fused cancellation");
    check(cases[0].ranking.sequential[0].id == 10 && cases[0].ranking.fourway[0].id == 20 &&
          cases[0].ranking.membership_changed && cases[0].difference.max_ulp == 1,
          "one ULP changes Top-1 membership with exact ID tie-breaking");
    const auto& r = cases[1].ranking;
    check(!r.membership_changed && r.id_order_changed && r.common_members == 3 &&
          r.sequential[0].id == 30 && r.fourway[0].id == 20, "full membership stable while ordering changes");
    check(!cases[2].ranking.membership_changed && !cases[2].ranking.id_order_changed &&
          cases[2].ranking.sequential[0].id == 10 && cases[2].ranking.sequential[1].id == 20,
          "exact ties always rank by ascending ID");
    check(cases[3].sequential_scores[0] == -std::ldexp(1.0F, -46) && cases[3].fourway_scores[0] == 0 &&
          cases[3].ranking.sequential[0].id == 10 && cases[3].ranking.fourway[0].id == 5 &&
          cases[3].difference.near_zero_reference == 2 && cases[3].difference.max_absolute == std::ldexp(1.0, -46),
          "cancellation residual, safe near-zero metrics and changed membership are reported");
    const std::array<vectorforge::RecordID, 3> ids{30, 10, 20};
    const std::array scores{-0.0F, 0.0F, 0.0F};
    const auto zeros = study::compare_rankings(ids, scores, scores, 99);
    check(zeros.sequential[0].id == 10 && zeros.sequential[2].id == 30, "signed zero comparator has no epsilon");
    check(study::compare_rankings(ids, scores, scores, 0).sequential.empty(), "K=0 comparison");
    expect_throw<std::invalid_argument>([&] { (void)study::compare_rankings(ids, {}, {}, 1); });
    const std::array<vectorforge::RecordID, 3> duplicate{10, 10, 20};
    expect_throw<std::invalid_argument>([&] { (void)study::compare_rankings(duplicate, scores, scores, 1); });
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

void configuration_timing_and_csv() {
    const auto defaults = study::parse_options({});
    check(defaults.sampling.iterations == 8, "balanced default sample count");
    for (const auto count : std::array<std::string_view, 3>{"3", "5", "7"}) {
        const std::array<std::string_view, 2> args{"--iterations", count};
        expect_throw<std::invalid_argument>([&] { (void)study::parse_options(args); });
    }
    const std::array<std::string_view, 6> args{"--rows", "17", "--dimension", "5", "--iterations", "4"};
    auto options = study::parse_options(args);
    options.sampling.smoke = true;
    options.sampling.warmup = 1;
    check(options.custom && options.custom->rows == 17 && options.custom->dimension == 5, "shape options reused");
    const study::Workload workload{"tiny, \"quoted\"", 17, 5};
    const auto run = study::run_workload(workload, options), repeat = study::run_workload(workload, options);
    check(run.dataset_checksum == repeat.dataset_checksum && run.difference.count == 17, "repeatable data and comparison counts");
    for (std::size_t i = 0; i < 2; ++i) {
        check(run.measurements[i].checksum == repeat.measurements[i].checksum && run.measurements[i].samples_ms.size() == 4 &&
              run.measurements[i].timing.min_ms > 0, "each changed-order kernel is independently repeatable/observable");
    }
    const bench::Metadata info{"GCC, \"test\"", "Debug", "-g", "cpu", "os", "line1\nline2", "UTC", true};
    std::ostringstream out;
    study::write_csv_header(out);
    study::write_csv(out, workload, options, info, run);
    const auto csv = read_csv(out.str());
    check(csv.size() == 3 && csv[0].size() == 37 && csv[1].size() == 37 && csv[2].size() == 37, "37-column comparison CSV");
    check(csv[1][0] == workload.id && csv[1][20] == info.compiler && csv[1][25] == info.environment,
          "quoted multiline metadata round-trips");
    check(csv[1][18] == csv[2][18] && csv[1][18] == "x86_64;SSE2;AVX;FMA" && csv[1][28] == "smoke",
          "same hardware ISA and smoke labels");
    check(csv[1][29] == "17" && std::stoull(csv[1][30]) == run.difference.equal_bits && csv[1][36] == csv[2][36],
          "pairwise numerical metrics exported unchanged on both rows");
    const auto empty = study::run_workload({"empty", 0, 3}, options);
    std::ostringstream empty_out;
    study::write_csv_header(empty_out);
    study::write_csv(empty_out, {"empty", 0, 3}, options, info, empty);
    check(read_csv(empty_out.str())[1][11].empty() && read_csv(empty_out.str())[1][31].empty(),
          "empty per-dot latency and equality rate remain undefined");
    std::ostringstream ranking;
    study::write_ranking_csv(ranking, info, study::ranking_study());
    const auto ranking_csv = read_csv(ranking.str());
    check(ranking_csv.size() == 5 && ranking_csv[0].size() == 32 && ranking_csv[1].size() == 32 &&
          ranking_csv[1][4] == "10" && ranking_csv[1][5] == "20" && ranking_csv[1][7] == "1" &&
          ranking_csv[1][31] == "numerical_analysis", "ranking IDs, score bits, membership and schema exported");
}

} // namespace

int main() {
    if (!sim::fma::hardware_supported()) { std::cout << "SKIP: AVX/FMA unavailable\n"; return 77; }
    try {
        tails_and_boundaries();
        numerical_reporting();
        ranking_sensitivity();
        configuration_timing_and_csv();
        std::cout << "PASS tails, changed-order oracle, numerical metrics, exact ranking sensitivity, balanced timing and CSV\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
