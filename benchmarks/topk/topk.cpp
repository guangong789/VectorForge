#include "topk.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace vectorforge::benchmark::topk {
namespace {

std::uint64_t number(std::string_view text) {
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument("expected a nonnegative decimal integer");
    }
    return value;
}

Distribution parse_distribution(std::string_view text) {
    if (text == "random") { return Distribution::random; }
    if (text == "ties") { return Distribution::ties; }
    if (text == "ascending") { return Distribution::ascending; }
    if (text == "descending") { return Distribution::descending; }
    throw std::invalid_argument("distribution must be random, ties, ascending or descending");
}

std::uint64_t fingerprint(std::span<const ScoredRecord> values) {
    std::uint64_t hash = 14'695'981'039'346'656'037ULL;
    for (const auto& value : values) {
        hash = (hash ^ value.id) * 1'099'511'628'211ULL;
        hash = (hash ^ std::bit_cast<std::uint32_t>(value.score)) * 1'099'511'628'211ULL;
    }
    return hash;
}

void exact_results(std::span<const ScoredRecord> actual, std::span<const ScoredRecord> expected) {
    if (actual.size() != expected.size()) { throw std::runtime_error("Top-K result count mismatch"); }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i].id != expected[i].id ||
            std::bit_cast<std::uint32_t>(actual[i].score) != std::bit_cast<std::uint32_t>(expected[i].score)) {
            throw std::runtime_error("Top-K ID, ordering or float32 score-bit mismatch");
        }
    }
}

std::string csv_quote(std::string_view value) {
    std::string quoted = "\"";
    for (const char c : value) {
        if (c == '"') { quoted += '"'; }
        quoted += c;
    }
    return quoted + '"';
}

} // namespace

std::string_view distribution_name(Distribution distribution) {
    switch (distribution) {
    case Distribution::random: return "random";
    case Distribution::ties: return "ties";
    case Distribution::ascending: return "ascending";
    case Distribution::descending: return "descending";
    }
    throw std::invalid_argument("unsupported score distribution");
}

void validate(const Workload& workload, const Options& options) {
    if (workload.id.empty() || options.iterations < 3 ||
        workload.candidates > std::vector<ScoredRecord>().max_size()) {
        throw std::invalid_argument("require an ID, iterations>=3 and an addressable candidate count");
    }
    (void)distribution_name(workload.distribution);
}

Options parse_options(std::span<const std::string_view> arguments) {
    Options options;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const auto name = arguments[i];
        if (name == "--help") { options.help = true; continue; }
        if (name == "--smoke") { options.smoke = true; continue; }
        if (i + 1 == arguments.size()) { throw std::invalid_argument("missing option value"); }
        const auto text = arguments[++i];
        if (name == "--distribution") {
            if (!options.custom) { options.custom = Workload{"custom"}; }
            options.custom->distribution = parse_distribution(text);
            continue;
        }
        const auto value = number(text);
        if (name == "--seed") { options.seed = value; continue; }
        if (value > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("configuration value exceeds size_t");
        }
        const auto size = static_cast<std::size_t>(value);
        if (name == "--warmup") { options.warmup = size; continue; }
        if (name == "--iterations") { options.iterations = size; continue; }
        if (!options.custom) { options.custom = Workload{"custom"}; }
        if (name == "--candidates") { options.custom->candidates = size; }
        else if (name == "--k") { options.custom->k = size; }
        else { throw std::invalid_argument("unknown Top-K option: " + std::string(name)); }
    }
    validate(options.custom.value_or(Workload{"default"}), options);
    if (options.smoke && options.custom) {
        throw std::invalid_argument("--smoke uses a fixed tiny candidate sequence");
    }
    return options;
}

std::vector<Workload> default_workloads() {
    return {
        {"zero_k", 10'000, 0, Distribution::random},
        {"k_one", 10'000, 1, Distribution::random},
        {"small_k", 10'000, 10, Distribution::random},
        {"near_m", 10'000, 9'000, Distribution::random},
        {"all_k", 10'000, 10'000, Distribution::random},
        {"beyond_m", 10'000, 10'001, Distribution::random},
        {"larger_m", 100'000, 10, Distribution::random},
        {"tie_heavy", 10'000, 10, Distribution::ties},
        {"ascending", 10'000, 10, Distribution::ascending},
        {"descending", 10'000, 10, Distribution::descending},
    };
}

std::vector<ScoredRecord> make_candidates(const Workload& workload, std::uint64_t seed) {
    validate(workload, Options{});
    std::mt19937_64 random(seed);
    std::vector<ScoredRecord> candidates;
    candidates.reserve(workload.candidates);
    for (std::size_t row = 0; row < workload.candidates; ++row) {
        const float score = workload.distribution == Distribution::ties
            ? static_cast<float>(static_cast<int>(random() % 8) - 4)
            : static_cast<float>(static_cast<int>(random() % 2'000'001) - 1'000'000) / 1024.0F;
        candidates.push_back({static_cast<RecordID>(workload.candidates - row), score});
    }
    if (workload.distribution == Distribution::ascending) {
        std::sort(candidates.begin(), candidates.end(), [](const ScoredRecord& a, const ScoredRecord& b) {
            return a.score < b.score || (a.score == b.score && a.id > b.id);
        });
    } else if (workload.distribution == Distribution::descending) {
        std::sort(candidates.begin(), candidates.end(), [](const ScoredRecord& a, const ScoredRecord& b) {
            return a.score > b.score || (a.score == b.score && a.id < b.id);
        });
    }
    return candidates;
}

void validate_candidates(std::span<const ScoredRecord> candidates) {
    std::unordered_set<RecordID> ids;
    for (const auto& candidate : candidates) {
        if (!std::isfinite(candidate.score) || !ids.insert(candidate.id).second) {
            throw std::invalid_argument("Top-K candidates require finite scores and unique IDs");
        }
    }
}

std::vector<ScoredRecord> full_sort(std::span<const ScoredRecord> candidates, std::size_t k) {
    if (k == 0) { return {}; }
    // Allocation/copy of M records is part of the measured operation.
    std::vector<ScoredRecord> result(candidates.begin(), candidates.end());
    std::sort(result.begin(), result.end(), [](const ScoredRecord& a, const ScoredRecord& b) {
        return a.score > b.score || (a.score == b.score && a.id < b.id);
    });
    result.resize(std::min(k, result.size()));
    return result;
}

std::vector<ScoredRecord> bounded_heap(std::span<const ScoredRecord> candidates, std::size_t k) {
    if (k == 0) { return {}; }
    const auto better = [](const ScoredRecord& a, const ScoredRecord& b) {
        return a.score > b.score || (a.score == b.score && a.id < b.id);
    };
    std::vector<ScoredRecord> result;
    for (const auto& candidate : candidates) {
        if (result.size() < k) {
            result.push_back(candidate);
            std::push_heap(result.begin(), result.end(), better);
        } else if (better(candidate, result.front())) {
            std::pop_heap(result.begin(), result.end(), better);
            result.back() = candidate;
            std::push_heap(result.begin(), result.end(), better);
        }
    }
    std::sort_heap(result.begin(), result.end(), better);
    return result;
}

Run run_workload(const Workload& workload, const Options& options) {
    validate(workload, options);
    const auto candidates = make_candidates(workload, options.seed);
    validate_candidates(candidates); // Includes non-finite/duplicate checks outside timing.
    const auto input_checksum = fingerprint(candidates);
    const auto expected = [&] {
        const auto sorted = full_sort(candidates, workload.k);
        // Keep only min(K,M) expected records resident; release the full M-sized
        // sorting workspace before warm-up/measurement. This copy is not timed.
        return std::vector<ScoredRecord>(sorted.begin(), sorted.end());
    }();
    exact_results(bounded_heap(candidates, workload.k), expected);
    const auto checksum = fingerprint(expected);
    Run run{input_checksum, {{
        {"full_sort", expected.size(), checksum, {}, {}},
        {"bounded_heap", expected.size(), checksum, {}, {}},
    }}};
    for (auto& result : run.measurements) { result.samples_ms.reserve(options.iterations); }
    auto sample = [&](std::size_t algorithm, bool measured) {
        const auto start = std::chrono::steady_clock::now();
        const auto result = algorithm == 0 ? full_sort(candidates, workload.k) : bounded_heap(candidates, workload.k);
        const auto finish = std::chrono::steady_clock::now();
        exact_results(result, expected);
        if (fingerprint(result) != checksum) { throw std::runtime_error("Top-K fingerprint mismatch"); }
        if (measured) {
            run.measurements[algorithm].samples_ms.push_back(
                std::chrono::duration<double, std::milli>(finish - start).count());
        }
    };
    for (std::size_t round = 0; round < options.warmup; ++round) {
        sample(round % 2, false);
        sample(1 - round % 2, false);
    }
    for (std::size_t round = 0; round < options.iterations; ++round) {
        sample(round % 2, true);
        sample(1 - round % 2, true);
    }
    if (fingerprint(candidates) != input_checksum) { throw std::runtime_error("candidate input was modified"); }
    for (auto& result : run.measurements) { result.timing = summarize(result.samples_ms); }
    return run;
}

void write_csv_header(std::ostream& output) {
    output << "workload,algorithm,m,k,k_over_m,distribution,warmup,iterations,seed,median_ms,min_ms,max_ms,"
              "returned_results,input_checksum,result_checksum,samples_ms,compiler,build_type,optimization_flags,"
              "cpu,os,environment,timestamp_utc,sanitized,mode,timing_boundary\n";
}

void write_csv(std::ostream& output, const Workload& workload, const Options& options,
               const Metadata& info, const Run& run) {
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto& result : run.measurements) {
        std::ostringstream samples;
        samples.imbue(std::locale::classic());
        samples << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (std::size_t i = 0; i < result.samples_ms.size(); ++i) {
            if (i != 0) { samples << ';'; }
            samples << result.samples_ms[i];
        }
        output << csv_quote(workload.id) << ',' << csv_quote(result.algorithm) << ',' << workload.candidates << ','
               << workload.k << ',';
        if (workload.candidates != 0) {
            output << static_cast<double>(workload.k) / static_cast<double>(workload.candidates);
        }
        output << ',' << csv_quote(distribution_name(workload.distribution)) << ',' << options.warmup << ','
               << options.iterations << ',' << options.seed << ',' << result.timing.median_ms << ','
               << result.timing.min_ms << ',' << result.timing.max_ms << ',' << result.returned << ','
               << run.input_checksum << ',' << result.checksum << ',' << csv_quote(samples.str()) << ','
               << csv_quote(info.compiler) << ',' << csv_quote(info.build_type) << ',' << csv_quote(info.optimization_flags) << ','
               << csv_quote(info.cpu) << ',' << csv_quote(info.os) << ',' << csv_quote(info.environment) << ','
               << csv_quote(info.timestamp_utc) << ',' << (info.sanitized ? 1 : 0) << ','
               << csv_quote(options.smoke ? "smoke" : "performance") << ','
               << "immutable_candidates_to_ordered_topk_including_allocation_and_copy\n";
    }
}

} // namespace vectorforge::benchmark::topk
