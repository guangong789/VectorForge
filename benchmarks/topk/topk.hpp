#pragma once

#include "benchmark.hpp"

namespace vectorforge::benchmark::topk {

enum class Distribution { random, ties, ascending, descending };

struct Workload {
    std::string id;
    std::size_t candidates = 10'000;
    std::size_t k = 10;
    Distribution distribution = Distribution::random;
};

struct Options {
    std::size_t warmup = 2;
    std::size_t iterations = 7;
    std::uint64_t seed = 0x56463036ULL;
    std::optional<Workload> custom;
    bool smoke = false;
    bool help = false;
};

struct Measurement {
    std::string algorithm;
    std::size_t returned;
    std::uint64_t checksum;
    std::vector<double> samples_ms;
    Timing timing;
};

struct Run {
    std::uint64_t input_checksum;
    std::array<Measurement, 2> measurements;
};

std::string_view distribution_name(Distribution distribution);
void validate(const Workload& workload, const Options& options);
Options parse_options(std::span<const std::string_view> arguments);
std::vector<Workload> default_workloads();
std::vector<ScoredRecord> make_candidates(const Workload& workload, std::uint64_t seed);
void validate_candidates(std::span<const ScoredRecord> candidates);

// Benchmark-local algorithms. Preconditions: finite scores and unique IDs,
// validated outside timing. Both borrow the same immutable input and return an
// owning ordered result. K=0 returns immediately without copying or scanning.
// full_sort includes allocation/copy of all M candidates and retains that vector's
// capacity on return. bounded_heap grows only with min(K,M), never reserves K,
// and includes its final in-place sort. Neither calls a production executor.
std::vector<ScoredRecord> full_sort(std::span<const ScoredRecord> candidates, std::size_t k);
std::vector<ScoredRecord> bounded_heap(std::span<const ScoredRecord> candidates, std::size_t k);

Run run_workload(const Workload& workload, const Options& options);
void write_csv_header(std::ostream& output);
void write_csv(std::ostream& output, const Workload& workload, const Options& options,
               const Metadata& info, const Run& run);

} // namespace vectorforge::benchmark::topk
