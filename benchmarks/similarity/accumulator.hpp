#pragma once

#include "fma_comparison.hpp"
#include "fourway_kernel.hpp"

namespace vectorforge::benchmark::accumulator {

using Workload = similarity::Workload;
using Options = similarity::Options;
inline constexpr double relative_floor = 1e-6;

struct Difference {
    std::size_t count = 0;
    std::size_t equal_bits = 0;
    std::size_t near_zero_reference = 0;
    double max_absolute = 0;
    double max_relative = 0;
    std::uint64_t max_ulp = 0;
};

// Finite scores only. Relative difference uses abs(a-b)/max(abs(reference),floor).
// ULP distance is distance in monotone binary32 keys, with signed zeros merged;
// bitwise equality still distinguishes their signs. Empty equality rate is undefined.
Difference differences(std::span<const float> reference, std::span<const float> experimental,
                       double floor = relative_floor);

struct Ranking {
    std::vector<ScoredRecord> sequential;
    std::vector<ScoredRecord> fourway;
    std::size_t common_members = 0;
    bool membership_changed = false;
    bool id_order_changed = false;
};
Ranking compare_rankings(std::span<const RecordID> ids, std::span<const float> sequential,
                         std::span<const float> fourway, std::size_t k);

struct RankingCase {
    std::string id;
    std::size_t dimension;
    std::size_t k;
    std::vector<RecordID> ids;
    std::vector<float> sequential_scores;
    std::vector<float> fourway_scores;
    Difference difference;
    Ranking ranking;
};
std::vector<RankingCase> ranking_study();

struct Measurement {
    std::string variant;
    std::string isa;
    std::string compile_options;
    std::uint64_t checksum;
    std::vector<double> samples_ms;
    Timing timing;
};
struct Run {
    std::uint64_t dataset_checksum;
    Difference difference;
    std::array<Measurement, 2> measurements;
};

// Eight measurements by default. Explicit measured counts must be even and >=4
// to balance first-run order exactly. Existing similarity CLI syntax otherwise.
Options parse_options(std::span<const std::string_view> arguments);
void validate(const Workload& workload, const Options& options);
Run run_workload(const Workload& workload, const Options& options);
void write_csv_header(std::ostream& output);
void write_csv(std::ostream& output, const Workload& workload, const Options& options,
               const Metadata& info, const Run& run);
void write_ranking_csv(std::ostream& output, const Metadata& info,
                       std::span<const RankingCase> cases);

} // namespace vectorforge::benchmark::accumulator
