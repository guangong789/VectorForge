#include "similarity.hpp"

#include "vectorforge/scalar_reference.hpp"

#include <bit>
#include <chrono>
#include <iomanip>
#include <limits>
#include <locale>
#include <random>
#include <sstream>
#include <stdexcept>

namespace vectorforge::benchmark::similarity {
namespace {

void mix(std::uint64_t& hash, std::uint64_t value) {
    hash = (hash ^ value) * 1'099'511'628'211ULL;
}

std::uint64_t fingerprint(std::span<const float> values) {
    std::uint64_t hash = 14'695'981'039'346'656'037ULL;
    for (const auto value : values) { mix(hash, std::bit_cast<std::uint32_t>(value)); }
    return hash;
}

// Use the unchanged production reference outside timing. Release these temporary
// owners, IDs and ranked records before warm-up; timed scans keep no second dataset.
std::vector<float> reference_scores(const Workload& workload, const Data& data) {
    std::vector<ColumnBatch::Column> columns;
    columns.emplace_back(std::vector<std::int32_t>(workload.rows, 0));
    const ColumnBatch batch(std::move(columns));
    const EmbeddingBuffer embeddings(workload.rows, workload.dimension, data.embeddings);
    std::vector<RecordID> ids(workload.rows);
    for (std::size_t row = 0; row < workload.rows; ++row) { ids[row] = static_cast<RecordID>(row); }
    const auto result = execute_scalar_reference(batch, embeddings, ids, data.query, {}, workload.rows);
    if (result.size() != workload.rows) { throw std::runtime_error("reference score count mismatch"); }
    std::vector<float> expected(workload.rows);
    for (const auto& record : result) { expected[static_cast<std::size_t>(record.id)] = record.score; }
    return expected;
}

void exact_scores(std::span<const float> actual, std::span<const float> expected) {
    if (actual.size() != expected.size()) { throw std::runtime_error("score count mismatch"); }
    for (std::size_t row = 0; row < actual.size(); ++row) {
        if (std::bit_cast<std::uint32_t>(actual[row]) != std::bit_cast<std::uint32_t>(expected[row])) {
            throw std::runtime_error("sequential FMA/reference float32 score-bit mismatch");
        }
    }
}

std::string quote(std::string_view value) {
    std::string result = "\"";
    for (const auto c : value) {
        if (c == '"') { result += '"'; }
        result += c;
    }
    return result + '"';
}

} // namespace

void validate(const Workload& workload, const Options& options) {
    benchmark::validate({workload.id, workload.rows, workload.dimension}, options.sampling);
}

Options parse_options(std::span<const std::string_view> arguments) {
    // Reuse strict decimal parsing and sampling conventions, but accept only
    // options relevant to this kernel (no selectivity, Top-K or batching knobs).
    bool rows_given = false;
    bool dimension_given = false;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const auto name = arguments[i];
        if (name == "--help" || name == "--smoke") { continue; }
        if (name != "--rows" && name != "--dimension" && name != "--warmup" &&
            name != "--iterations" && name != "--seed") {
            throw std::invalid_argument("unknown similarity option: " + std::string(name));
        }
        rows_given = rows_given || name == "--rows";
        dimension_given = dimension_given || name == "--dimension";
        if (++i == arguments.size()) { throw std::invalid_argument("missing similarity option value"); }
    }
    std::vector<std::string_view> forwarded(arguments.begin(), arguments.end());
    if (dimension_given && !rows_given) {
        forwarded.insert(forwarded.end(), {"--rows", "1024"});
    }
    Options options{benchmark::parse_options(forwarded), {}};
    if (options.sampling.custom) {
        const auto& custom = *options.sampling.custom;
        options.custom = Workload{"custom", custom.rows, custom.dimension};
        options.sampling.custom.reset();
    }
    validate(options.custom.value_or(Workload{"default"}), options);
    return options;
}

std::vector<Workload> default_workloads() {
    return {{"small_d32", 1024, 32}, {"larger_d32", 8192, 32},
            {"small_d128", 1024, 128}, {"larger_d128", 8192, 128},
            {"small_d768", 1024, 768}, {"larger_d768", 8192, 768}};
}

Data make_data(const Workload& workload, std::uint64_t seed) {
    validate(workload, {});
    std::mt19937_64 random(seed);
    auto next = [&] {
        return static_cast<float>(static_cast<int>(random() % 2049) - 1024) / 1024.0F;
    };
    Data data{std::vector<float>(workload.dimension),
              std::vector<float>(workload.rows * workload.dimension), 14'695'981'039'346'656'037ULL};
    for (auto& value : data.query) { value = next(); }
    for (auto& value : data.embeddings) { value = next(); }
    mix(data.checksum, workload.rows);
    mix(data.checksum, workload.dimension);
    mix(data.checksum, fingerprint(data.query));
    mix(data.checksum, fingerprint(data.embeddings));
    return data;
}

Run run_workload(const Workload& workload, const Options& options) {
    validate(workload, options);
    const auto data = make_data(workload, options.sampling.seed);
    const auto expected = reference_scores(workload, data);
    std::vector<float> scores(workload.rows);
    sequential_fma_scores(data.query, data.embeddings, scores);
    exact_scores(scores, expected);
    Run run{data.checksum, fingerprint(expected), {}, {}};
    run.samples_ms.reserve(options.sampling.iterations);
    auto consume = [&](bool measured) {
        const auto start = std::chrono::steady_clock::now();
        sequential_fma_scores(data.query, data.embeddings, scores);
        const auto finish = std::chrono::steady_clock::now();
        // Consume every row after EVERY call, including warm-up. The kernel is a
        // separate TU; output writes are observable, not a discarded scalar sum.
        exact_scores(scores, expected);
        if (fingerprint(scores) != run.result_checksum) { throw std::runtime_error("score checksum mismatch"); }
        if (measured) {
            run.samples_ms.push_back(std::chrono::duration<double, std::milli>(finish - start).count());
        }
    };
    for (std::size_t i = 0; i < options.sampling.warmup; ++i) { consume(false); }
    for (std::size_t i = 0; i < options.sampling.iterations; ++i) { consume(true); }
    run.timing = benchmark::summarize(run.samples_ms);
    return run;
}

void write_csv_header(std::ostream& output) {
    output << "workload,kernel,n,d,embedding_bytes,warmup,iterations,seed,median_ms,min_ms,max_ms,"
              "ns_per_dot,dots_per_second,dataset_checksum,result_checksum,samples_ms,timing_scope,"
              "kernel_isa,compiler,build_type,optimization_flags,cpu,os,environment,timestamp_utc,sanitized,mode\n";
}

void write_csv(std::ostream& output, const Workload& workload, const Options& options,
               const Metadata& info, const Run& run) {
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::ostringstream samples;
    samples.imbue(std::locale::classic());
    samples << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (std::size_t i = 0; i < run.samples_ms.size(); ++i) {
        if (i != 0) { samples << ';'; }
        samples << run.samples_ms[i];
    }
    output << quote(workload.id) << ",\"sequential_float32_fma\"," << workload.rows << ','
           << workload.dimension << ',' << workload.rows * workload.dimension * sizeof(float) << ','
           << options.sampling.warmup << ',' << options.sampling.iterations << ',' << options.sampling.seed << ','
           << run.timing.median_ms << ',' << run.timing.min_ms << ',' << run.timing.max_ms << ',';
    if (workload.rows != 0) { output << run.timing.median_ms * 1e6 / static_cast<double>(workload.rows); }
    output << ',' << static_cast<double>(workload.rows) * 1000 / run.timing.median_ms << ','
           << run.dataset_checksum << ',' << run.result_checksum << ',' << quote(samples.str()) << ','
           << "\"one_row_major_scan_shape_checks_finite_checks_output_stores\"," << quote(kernel_isa()) << ','
           << quote(info.compiler) << ',' << quote(info.build_type) << ',' << quote(info.optimization_flags) << ','
           << quote(info.cpu) << ',' << quote(info.os) << ',' << quote(info.environment) << ','
           << quote(info.timestamp_utc) << ',' << (info.sanitized ? 1 : 0) << ','
           << quote(options.sampling.smoke ? "smoke" : "performance") << '\n';
}

} // namespace vectorforge::benchmark::similarity
