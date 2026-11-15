#pragma once

#include "similarity.hpp"

namespace vectorforge::benchmark::similarity {

namespace generic_target {
void sequential_fma_scores(std::span<const float>, std::span<const float>, std::span<float>);
std::string_view kernel_isa();
}
namespace hardware_target {
void sequential_fma_scores(std::span<const float>, std::span<const float>, std::span<float>);
std::string_view kernel_isa();
}

namespace fma {

// This guard is compiled for the baseline target, never with AVX/FMA options.
// No function from hardware_target may be called before it succeeds.
bool hardware_supported();

struct Measurement {
    std::string variant;
    std::string isa;
    std::string compile_options;
    std::vector<double> samples_ms;
    Timing timing;
};

struct Run {
    std::uint64_t dataset_checksum;
    std::uint64_t result_checksum;
    std::array<Measurement, 2> measurements;
};

// Same inputs, preallocated output, exact verification outside timing. Every
// round runs both targets; first target alternates in warm-up and measurement.
Run run_workload(const Workload& workload, const Options& options);
void write_csv_header(std::ostream& output);
void write_csv(std::ostream& output, const Workload& workload, const Options& options,
               const Metadata& info, const Run& run);

} // namespace fma
} // namespace vectorforge::benchmark::similarity
