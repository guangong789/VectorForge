#pragma once

#include "benchmark.hpp"
#include "similarity_kernel.hpp"

namespace vectorforge::benchmark::similarity {

struct Workload {
    std::string id;
    std::size_t rows = 1024;
    std::size_t dimension = 32;
};

struct Options {
    benchmark::Options sampling;
    std::optional<Workload> custom;
};

struct Data {
    std::vector<float> query;
    std::vector<float> embeddings;
    std::uint64_t checksum;
};

struct Run {
    std::uint64_t dataset_checksum;
    std::uint64_t result_checksum;
    std::vector<double> samples_ms;
    Timing timing;
};

void validate(const Workload& workload, const Options& options);
Options parse_options(std::span<const std::string_view> arguments);
std::vector<Workload> default_workloads();
Data make_data(const Workload& workload, std::uint64_t seed);
Run run_workload(const Workload& workload, const Options& options);
void write_csv_header(std::ostream& output);
void write_csv(std::ostream& output, const Workload& workload, const Options& options,
               const Metadata& info, const Run& run);

} // namespace vectorforge::benchmark::similarity
