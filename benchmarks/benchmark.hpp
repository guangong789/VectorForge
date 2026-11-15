#pragma once

#include "vectorforge/column_batch.hpp"
#include "vectorforge/embedding_buffer.hpp"
#include "vectorforge/query_types.hpp"

#include <array>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Benchmark-only support, separate from the engine's public API.
namespace vectorforge::benchmark {

struct Workload {
    std::string id;
    std::size_t rows = 10'000;
    std::size_t dimension = 32;
    unsigned selectivity_percent = 10;
    std::size_t k = 10;
    std::size_t batch_size = 256;
};

struct Options {
    std::size_t warmup = 2;
    std::size_t iterations = 7;
    std::uint64_t seed = 0x56463035ULL;
    std::optional<Workload> custom;
    bool smoke = false;
    bool help = false;
};

struct Dataset {
    ColumnBatch columns;
    EmbeddingBuffer embeddings;
    std::vector<RecordID> ids;
    std::vector<float> query;
    std::size_t qualifying;
    std::uint64_t checksum;
};

struct Timing {
    double median_ms;
    double min_ms;
    double max_ms;
};

struct Measurement {
    std::string executor;
    std::size_t batch_size;
    std::size_t returned;
    std::uint64_t checksum;
    std::vector<double> samples_ms;
    Timing timing;
};

struct Run {
    std::size_t qualifying;
    std::uint64_t dataset_checksum;
    std::array<Measurement, 2> measurements;
};

struct Metadata {
    std::string compiler;
    std::string build_type;
    std::string optimization_flags;
    std::string cpu;
    std::string os;
    std::string environment;
    std::string timestamp_utc;
    bool sanitized;
};

void validate(const Workload& workload, const Options& options);
Options parse_options(std::span<const std::string_view> arguments);
std::vector<Workload> default_workloads();
Dataset make_dataset(const Workload& workload, std::uint64_t seed);
Timing summarize(std::span<const double> samples_ms);
Metadata metadata();
void require_performance_build(const Metadata& info, bool smoke);
Run run_workload(const Workload& workload, const Options& options);
void write_csv_header(std::ostream& output);
void write_csv(std::ostream& output, const Workload& workload, const Options& options,
               const Metadata& info, const Run& run);

} // namespace vectorforge::benchmark
