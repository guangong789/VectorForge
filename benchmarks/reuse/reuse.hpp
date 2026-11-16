#pragma once

#include "benchmark.hpp"

namespace vectorforge::benchmark::reuse {

struct Workload {
    benchmark::Workload query;
    std::vector<std::size_t> reuse_counts{1, 10, 100};
};

struct Options {
    benchmark::Options sampling;
    std::optional<std::size_t> custom_reuse;
};

struct Samples {
    std::vector<double> values_ms;
    Timing timing;
};

struct Run {
    std::size_t qualifying;
    std::size_t returned;
    std::uint64_t dataset_checksum;
    std::uint64_t result_checksum;
    Samples admission;
    Samples legacy_query;
    Samples trusted_query;
};

struct Model {
    double legacy_total_ms;
    double trusted_total_ms;
    std::optional<std::uint64_t> break_even_reuse;
};

void validate(const Workload& workload, const Options& options);
Options parse_options(std::span<const std::string_view> arguments);
std::vector<Workload> default_workloads();

// Model only, not a timed workflow: R*legacy and admission+R*trusted, using
// measured component medians. Equality is not a win. A finite positive threshold
// is floor(admission/(legacy-trusted))+1; no threshold if trusted >= legacy.
// Invalid/non-finite latencies or R=0 are rejected; unrepresentable results throw.
Model model(double admission_ms, double legacy_ms, double trusted_ms, std::size_t reuse_count);

Run run_workload(const Workload& workload, const Options& options);
void write_csv_header(std::ostream& output);
void write_csv(std::ostream& output, const Workload& workload, const Options& options,
               const Metadata& info, const Run& run);

} // namespace vectorforge::benchmark::reuse
