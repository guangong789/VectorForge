#include "similarity/fma_comparison.hpp"

#include "vectorforge/scalar_reference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
namespace vf = vectorforge;
namespace sim = vf::benchmark::similarity;
namespace fma = sim::fma;

using Kernel = void (*)(std::span<const float>, std::span<const float>, std::span<float>);
constexpr std::array<Kernel, 3> kernels{sim::sequential_fma_scores,
    sim::generic_target::sequential_fma_scores, sim::hardware_target::sequential_fma_scores};

void check(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <typename Error, typename Function>
void expect_throw(Function&& function) {
    try { function(); }
    catch (const Error&) { return; }
    throw std::runtime_error("expected exception was not thrown");
}

std::vector<float> compare_to_production(std::span<const float> query, std::span<const float> data) {
    const auto rows = data.size() / query.size();
    std::vector<vf::ColumnBatch::Column> columns;
    columns.emplace_back(std::vector<std::int32_t>(rows));
    const vf::ColumnBatch batch(std::move(columns));
    const vf::EmbeddingBuffer embeddings(rows, query.size(), std::vector<float>(data.begin(), data.end()));
    std::vector<vf::RecordID> ids(rows);
    for (std::size_t i = 0; i < rows; ++i) { ids[i] = static_cast<vf::RecordID>(i); }
    const auto records = vf::execute_scalar_reference(batch, embeddings, ids, query, {}, rows);
    std::vector<float> expected(rows);
    for (const auto& record : records) { expected[static_cast<std::size_t>(record.id)] = record.score; }
    std::vector<float> scores(rows);
    for (const auto kernel : kernels) {
        for (int repetition = 0; repetition < 5; ++repetition) {
            std::fill(scores.begin(), scores.end(), std::numeric_limits<float>::quiet_NaN());
            kernel(query, data, scores);
            for (std::size_t row = 0; row < rows; ++row) {
                check(std::bit_cast<std::uint32_t>(scores[row]) == std::bit_cast<std::uint32_t>(expected[row]),
                      "target/original/production score bits must match on every repeated execution");
            }
        }
    }
    return expected;
}

void numerical_contract() {
    for (const auto d : std::array<std::size_t, 9>{1, 2, 3, 7, 32, 127, 128, 129, 768}) {
        for (const auto seed : std::array<std::uint64_t, 3>{0, 42, 0x56463035ULL}) {
            const auto data = sim::make_data({"finite", 17, d}, seed);
            (void)compare_to_production(data.query, data.embeddings);
        }
    }
    const float a = std::ldexp(1.0F, -23);
    const std::array fused_query{-1.0F, 1.0F + a};
    const std::array fused_data{1.0F, 1.0F - a};
    check(compare_to_production(fused_query, fused_data)[0] == -std::ldexp(1.0F, -46),
          "fused cancellation residual retained");
    const std::array ones{1.0F, 1.0F, 1.0F};
    const std::array cancellation{16777216.0F, 1.0F, -16777216.0F, 16777216.0F, -16777216.0F, 1.0F};
    check(compare_to_production(ones, cancellation) == std::vector{0.0F, 1.0F},
          "feature order and intermediate rounding remain sequential");
    const std::array single_one{1.0F};
    const float denorm = std::numeric_limits<float>::denorm_min();
    const std::array tiny_data{denorm, -denorm, std::numeric_limits<float>::min()};
    const auto tiny = compare_to_production(single_one, tiny_data);
    check(std::bit_cast<std::uint32_t>(tiny[0]) == 1 && std::bit_cast<std::uint32_t>(tiny[1]) == 0x80000001U,
          "positive and negative subnormals preserved");
    const std::array half{0.5F};
    const auto underflow = compare_to_production(half, tiny_data);
    check(std::bit_cast<std::uint32_t>(underflow[0]) == 0 && underflow[2] == std::numeric_limits<float>::min() / 2,
          "gradual underflow and ties-to-even zero boundary");
    const std::array signed_zeros{-0.0F, 0.0F};
    const auto zeros = compare_to_production(single_one, signed_zeros);
    check(std::bit_cast<std::uint32_t>(zeros[0]) == 0 && std::bit_cast<std::uint32_t>(zeros[1]) == 0,
          "positive-zero accumulator sign");
    const float largest = std::numeric_limits<float>::max();
    const std::array large_query{-largest, largest};
    const std::array large_data{1.0F, 2.0F};
    check(compare_to_production(large_query, large_data)[0] == largest, "fused product does not overflow prematurely");
    (void)compare_to_production(single_one, {});

    const std::array overflow_query{2.0F, -2.0F};
    const std::array overflow_data{largest, largest};
    std::array<float, 1> score{};
    for (const auto kernel : kernels) {
        expect_throw<std::overflow_error>([&] { kernel(overflow_query, overflow_data, score); });
        expect_throw<std::invalid_argument>([&] { kernel({}, {}, {}); });
        expect_throw<std::invalid_argument>([&] { kernel(single_one, overflow_data, score); });
        expect_throw<std::invalid_argument>([&] { kernel(ones, overflow_data, score); });
    }
    std::vector<vf::ColumnBatch::Column> columns;
    columns.emplace_back(std::vector<std::int32_t>{0});
    const vf::ColumnBatch batch(std::move(columns));
    const vf::EmbeddingBuffer embeddings(1, 2, {largest, largest});
    const std::array<vf::RecordID, 1> ids{0};
    expect_throw<std::overflow_error>([&] {
        (void)vf::execute_scalar_reference(batch, embeddings, ids, overflow_query, {}, 1);
    });
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

void sampling_and_csv() {
    sim::Options options;
    options.sampling.smoke = true;
    options.sampling.warmup = 1;
    options.sampling.iterations = 4;
    const sim::Workload workload{"tiny, \"quoted\"", 17, 5};
    const auto run = fma::run_workload(workload, options);
    const auto repeat = fma::run_workload(workload, options);
    check(run.dataset_checksum == repeat.dataset_checksum && run.result_checksum == repeat.result_checksum,
          "repeated target comparisons reproduce inputs and score bits");
    const vf::benchmark::Metadata info{"GCC, \"test\"", "Debug", "-g", "cpu", "os", "line1\nline2", "UTC", true};
    std::ostringstream output;
    fma::write_csv_header(output);
    fma::write_csv(output, workload, options, info, run);
    const auto csv = read_csv(output.str());
    check(csv.size() == 3 && csv[0].size() == 29 && csv[1].size() == 29 && csv[2].size() == 29,
          "two target rows in 29-column CSV");
    check(csv[1][0] == workload.id && csv[1][20] == info.compiler && csv[1][25] == info.environment,
          "escaped quoted multiline CSV metadata");
    check(csv[1][13] == csv[2][13] && csv[1][14] == csv[2][14], "same input/output fingerprints for both targets");
    check(csv[1][18] == "x86_64;SSE2" && csv[2][18] == "x86_64;SSE2;AVX;FMA", "actual kernel ISA macros");
    check(csv[1][19].find("-mno-fma") != std::string::npos && csv[2][19].find("-mfma") != std::string::npos,
          "variant-specific flags recorded");
    check(csv[1][17] == "alternate_first_each_round_generic_first_round0" && csv[2][28] == "smoke",
          "order policy and smoke classification recorded");
    for (const auto& m : run.measurements) {
        check(m.samples_ms.size() == 4 && m.timing.min_ms > 0 && m.timing.min_ms <= m.timing.median_ms &&
              m.timing.median_ms <= m.timing.max_ms, "four alternating samples each, positive variability");
    }
    const auto empty = fma::run_workload({"empty", 0, 3}, options);
    std::ostringstream empty_output;
    fma::write_csv_header(empty_output);
    fma::write_csv(empty_output, {"empty", 0, 3}, options, info, empty);
    check(read_csv(empty_output.str())[1][11].empty(), "empty case has no per-dot latency");
    expect_throw<std::invalid_argument>([&] { (void)fma::run_workload({"bad", 1, 0}, options); });
}

} // namespace

int main() {
    if (!fma::hardware_supported()) {
        std::cout << "SKIP: AVX/FMA unavailable; hardware kernel was not called.\n";
        return 77;
    }
    try {
        numerical_contract();
        sampling_and_csv();
        std::cout << "PASS repeated bitwise target/production equivalence, boundaries and comparison CSV\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
