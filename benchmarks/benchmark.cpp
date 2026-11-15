#include "benchmark.hpp"

#include "benchmark_build_info.hpp"
#include "vectorforge/batched_executor.hpp"
#include "vectorforge/scalar_reference.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <sys/utsname.h>
#endif

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0) || defined(_M_FP_FAST)
#error "VectorForge benchmarks require the no-fast-math numerical contract"
#endif

namespace vectorforge::benchmark {
namespace {

std::uint64_t unsigned_number(std::string_view value) {
    std::uint64_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("expected a nonnegative decimal integer");
    }
    return result;
}

std::size_t checked_size(std::uint64_t value) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("configuration value exceeds size_t");
    }
    return static_cast<std::size_t>(value);
}

void mix(std::uint64_t& hash, std::uint64_t value) {
    // A reproducible fingerprint, not a cryptographic hash.
    hash = (hash ^ value) * 1'099'511'628'211ULL;
}

std::uint64_t result_checksum(std::span<const ScoredRecord> result) {
    std::uint64_t hash = 14'695'981'039'346'656'037ULL;
    for (const auto& record : result) {
        mix(hash, record.id);
        mix(hash, std::bit_cast<std::uint32_t>(record.score));
    }
    return hash;
}

void require_equal(std::span<const ScoredRecord> actual, std::span<const ScoredRecord> expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("reference/batched result count mismatch");
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i].id != expected[i].id ||
            std::bit_cast<std::uint32_t>(actual[i].score) != std::bit_cast<std::uint32_t>(expected[i].score)) {
            throw std::runtime_error("reference/batched ID, ordering or float32 score mismatch");
        }
    }
}

std::string first_field(const char* path, std::string_view key) {
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        if (line.starts_with(key)) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                const auto start = line.find_first_not_of(" \t", colon + 1);
                return start == std::string::npos ? "" : line.substr(start);
            }
        }
    }
    return "unavailable";
}

std::string csv_quote(std::string_view value) {
    std::string result = "\"";
    for (const char character : value) {
        if (character == '"') {
            result += '"';
        }
        result += character;
    }
    return result + '"';
}

} // namespace

void validate(const Workload& workload, const Options& options) {
    if (workload.id.empty() || workload.dimension == 0 || workload.batch_size == 0 ||
        workload.selectivity_percent > 100 || options.iterations < 3) {
        throw std::invalid_argument("require a workload ID, D>0, B>0, selectivity in [0,100], and iterations>=3");
    }
    if (workload.dimension > std::vector<float>().max_size() ||
        workload.rows > std::numeric_limits<std::size_t>::max() / workload.dimension ||
        workload.rows * workload.dimension > std::vector<float>().max_size() ||
        workload.rows > std::vector<RecordID>().max_size()) {
        throw std::invalid_argument("dataset shape exceeds addressable vector storage");
    }
}

Options parse_options(std::span<const std::string_view> arguments) {
    Options options;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const auto name = arguments[i];
        if (name == "--help") { options.help = true; continue; }
        if (name == "--smoke") { options.smoke = true; continue; }
        if (i + 1 == arguments.size()) {
            throw std::invalid_argument("missing value for option " + std::string(name));
        }
        const auto value = unsigned_number(arguments[++i]);
        if (name == "--seed") { options.seed = value; continue; }
        const auto size = checked_size(value);
        if (name == "--warmup") { options.warmup = size; continue; }
        if (name == "--iterations") { options.iterations = size; continue; }
        if (!options.custom) {
            options.custom = Workload{"custom"};
        }
        if (name == "--rows") { options.custom->rows = size; }
        else if (name == "--dimension") { options.custom->dimension = size; }
        else if (name == "--k") { options.custom->k = size; }
        else if (name == "--batch-size") { options.custom->batch_size = size; }
        else if (name == "--selectivity" && value <= 100) {
            options.custom->selectivity_percent = static_cast<unsigned>(value);
        } else {
            throw std::invalid_argument("unknown option or invalid selectivity: " + std::string(name));
        }
    }
    validate(options.custom.value_or(Workload{"default"}), options);
    if (options.smoke && options.custom) {
        throw std::invalid_argument("--smoke uses a fixed tiny workload; do not combine with workload options");
    }
    return options;
}

std::vector<Workload> default_workloads() {
    return {
        {"low_selectivity", 10'000, 32, 10, 10, 256},
        {"high_selectivity", 10'000, 32, 100, 10, 256},
        {"large_k", 10'000, 32, 100, 10'000, 256},
        {"higher_dimension", 10'000, 128, 100, 10, 256},
        {"larger_dataset", 50'000, 32, 100, 10, 256},
        {"larger_low_selectivity", 50'000, 32, 1, 10, 256},
        {"small_batch", 10'000, 32, 100, 10, 32},
        {"large_batch", 10'000, 32, 100, 10, 2048},
    };
}

Dataset make_dataset(const Workload& workload, std::uint64_t seed) {
    validate(workload, Options{});
    const auto rows = workload.rows;
    const auto dimension = workload.dimension;
    const auto percent = workload.selectivity_percent;
    const auto qualifying = (rows / 100) * percent + ((rows % 100) * percent) / 100;
    std::mt19937_64 random(seed);
    auto next_float = [&] {
        return static_cast<float>(static_cast<int>(random() % 2049) - 1024) / 1024.0F;
    };
    std::vector<float> query(dimension);
    for (auto& value : query) { value = next_float(); }
    std::vector<float> embeddings(rows * dimension);
    for (auto& value : embeddings) { value = next_float(); }
    std::vector<std::int32_t> flags(rows, 0);
    std::fill_n(flags.begin(), qualifying, 1);
    std::mt19937_64 filter_random(seed ^ 0x9E3779B97F4A7C15ULL);
    for (std::size_t remaining = rows; remaining > 1; --remaining) {
        const auto index = static_cast<std::size_t>(filter_random() % remaining);
        std::swap(flags[remaining - 1], flags[index]);
    }
    std::vector<RecordID> ids(rows);
    std::uint64_t checksum = 14'695'981'039'346'656'037ULL;
    mix(checksum, rows);
    mix(checksum, dimension);
    for (std::size_t row = 0; row < rows; ++row) {
        ids[row] = static_cast<RecordID>(rows - row);
        mix(checksum, ids[row]);
        mix(checksum, static_cast<std::uint64_t>(flags[row]));
    }
    for (const auto value : query) { mix(checksum, std::bit_cast<std::uint32_t>(value)); }
    for (const auto value : embeddings) { mix(checksum, std::bit_cast<std::uint32_t>(value)); }
    std::vector<ColumnBatch::Column> columns;
    columns.emplace_back(std::move(flags));
    return {ColumnBatch(std::move(columns)), EmbeddingBuffer(rows, dimension, std::move(embeddings)),
            std::move(ids), std::move(query), qualifying, checksum};
}

Timing summarize(std::span<const double> samples_ms) {
    if (samples_ms.empty()) {
        throw std::invalid_argument("cannot summarize empty timing samples");
    }
    std::vector<double> sorted(samples_ms.begin(), samples_ms.end());
    for (const auto value : sorted) {
        if (!std::isfinite(value) || value <= 0) {
            throw std::invalid_argument("timing samples must be positive and finite");
        }
    }
    std::sort(sorted.begin(), sorted.end());
    const auto middle = sorted.size() / 2;
    const double median = sorted.size() % 2 == 0
        ? sorted[middle - 1] / 2 + sorted[middle] / 2 : sorted[middle];
    return {median, sorted.front(), sorted.back()};
}

Metadata metadata() {
    std::string os = "unavailable";
    std::string environment = "unavailable";
#if defined(__linux__)
    utsname system{};
    if (uname(&system) == 0) {
        os = std::string(system.sysname) + " " + system.release + " " + system.machine;
    }
    std::ifstream init_file("/proc/1/comm");
    std::string init;
    std::getline(init_file, init);
    environment = "pid1=" + init + ";seccomp=" + first_field("/proc/self/status", "Seccomp:") +
        ";no_new_privs=" + first_field("/proc/self/status", "NoNewPrivs:") +
        ";logical_cpus=" + std::to_string(std::thread::hardware_concurrency());
    if (const auto* distro = std::getenv("WSL_DISTRO_NAME")) {
        environment += ";wsl_distro=" + std::string(distro);
    }
#endif
    const auto now = std::time(nullptr);
    std::ostringstream timestamp;
    if (const auto* utc = std::gmtime(&now)) { timestamp << std::put_time(utc, "%Y-%m-%dT%H:%M:%SZ"); }
    return {build_info::compiler, build_info::build_type, build_info::optimization_flags,
            first_field("/proc/cpuinfo", "model name"), os, environment, timestamp.str(), build_info::sanitized};
}

void require_performance_build(const Metadata& info, bool smoke) {
    if (!smoke && (info.sanitized || (info.build_type != "Release" && info.build_type != "RelWithDebInfo" &&
                                    info.build_type != "MinSizeRel"))) {
        throw std::invalid_argument("performance runs require an optimized unsanitized build; use --smoke for correctness only");
    }
}

Run run_workload(const Workload& workload, const Options& options) {
    validate(workload, options);
    const auto dataset = make_dataset(workload, options.seed);
    const std::array predicates{Predicate{0, Comparison::equal, std::int32_t{1}}};
    auto execute = [&](std::size_t executor) {
        if (executor == 0) {
            return execute_scalar_reference(dataset.columns, dataset.embeddings, dataset.ids,
                                             dataset.query, predicates, workload.k);
        }
        return execute_batched(dataset.columns, dataset.embeddings, dataset.ids, dataset.query,
                               predicates, workload.k, workload.batch_size);
    };
    // Establish exact correctness before warm-up or measurement, with the same
    // owners and query. Also consume/check every later result outside timing.
    const auto expected = execute(0);
    require_equal(execute(1), expected);
    const auto checksum = result_checksum(expected);
    Run run{dataset.qualifying, dataset.checksum, {{
        {"reference_full_sort", 0, expected.size(), checksum, {}, {}},
        {"batched_bounded_heap", workload.batch_size, expected.size(), checksum, {}, {}},
    }}};
    for (auto& measurement : run.measurements) { measurement.samples_ms.reserve(options.iterations); }
    auto consume = [&](std::size_t executor, bool measured) {
        const auto start = std::chrono::steady_clock::now();
        const auto result = execute(executor);
        const auto finish = std::chrono::steady_clock::now();
        require_equal(result, expected);
        if (result_checksum(result) != checksum) { throw std::runtime_error("result fingerprint mismatch"); }
        if (measured) {
            run.measurements[executor].samples_ms.push_back(
                std::chrono::duration<double, std::milli>(finish - start).count());
        }
    };
    for (std::size_t round = 0; round < options.warmup; ++round) {
        consume(round % 2, false);
        consume(1 - round % 2, false);
    }
    for (std::size_t round = 0; round < options.iterations; ++round) {
        consume(round % 2, true);
        consume(1 - round % 2, true);
    }
    for (auto& measurement : run.measurements) { measurement.timing = summarize(measurement.samples_ms); }
    return run;
}

void write_csv_header(std::ostream& output) {
    output << "workload,executor,n,d,k,batch_size,selectivity_percent,actual_selectivity,qualifying_candidates,"
              "returned_results,warmup,iterations,seed,median_ms,min_ms,max_ms,rows_per_second,result_checksum,"
              "dataset_checksum,samples_ms,compiler,build_type,optimization_flags,cpu,os,environment,timestamp_utc,"
              "sanitized,mode\n";
}

void write_csv(std::ostream& output, const Workload& workload, const Options& options,
               const Metadata& info, const Run& run) {
    // Formatting is deliberately after both executors' measured iterations.
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    const double selectivity = workload.rows == 0 ? 0 : static_cast<double>(run.qualifying) / static_cast<double>(workload.rows);
    for (const auto& measurement : run.measurements) {
        std::ostringstream samples;
        samples.imbue(std::locale::classic());
        samples << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (std::size_t i = 0; i < measurement.samples_ms.size(); ++i) {
            if (i != 0) { samples << ';'; }
            samples << measurement.samples_ms[i];
        }
        output << csv_quote(workload.id) << ',' << csv_quote(measurement.executor) << ',' << workload.rows << ','
               << workload.dimension << ',' << workload.k << ',' << measurement.batch_size << ','
               << workload.selectivity_percent << ',' << selectivity << ',' << run.qualifying << ','
               << measurement.returned << ',' << options.warmup << ',' << options.iterations << ',' << options.seed << ','
               << measurement.timing.median_ms << ',' << measurement.timing.min_ms << ',' << measurement.timing.max_ms << ','
               << static_cast<double>(workload.rows) * 1000 / measurement.timing.median_ms << ',' << measurement.checksum << ','
               << run.dataset_checksum << ',' << csv_quote(samples.str()) << ',' << csv_quote(info.compiler) << ','
               << csv_quote(info.build_type) << ',' << csv_quote(info.optimization_flags) << ',' << csv_quote(info.cpu) << ','
               << csv_quote(info.os) << ',' << csv_quote(info.environment) << ',' << csv_quote(info.timestamp_utc) << ','
               << (info.sanitized ? 1 : 0) << ',' << csv_quote(options.smoke ? "smoke" : "performance") << '\n';
    }
}

} // namespace vectorforge::benchmark
