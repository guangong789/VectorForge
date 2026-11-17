#include <vectorforge/batched_executor.hpp>
#include <vectorforge/column_batch.hpp>
#include <vectorforge/embedding_buffer.hpp>
#include <vectorforge/filtered_count.hpp>
#include <vectorforge/filtered_sum.hpp>
#include <vectorforge/query_types.hpp>
#include <vectorforge/scalar_reference.hpp>
#include <vectorforge/validated_dataset.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {
using namespace vectorforge;
constexpr std::uint64_t seed = 0x56463112CULL;
constexpr std::size_t dimension = 128, top_k = 20, batch_size = 1024, query_count = 8;
constexpr std::size_t tenant_column = 0, category_column = 1, updated_column = 2, tokens_column = 3;
constexpr std::int32_t tenant = 7, category = 2;
constexpr std::int64_t cutoff = 1'700'000'000;
constexpr std::size_t memory_budget = 512ULL * 1024 * 1024;
struct Cell { std::string_view name; std::size_t rows, matches; };
constexpr std::array cells{Cell{"N10K-S1", 10'000, 100}, Cell{"N10K-S90", 10'000, 9'000},
                           Cell{"N100K-S1", 100'000, 1'000}, Cell{"N100K-S90", 100'000, 90'000}};
const std::array predicates{Predicate{tenant_column, Comparison::equal, tenant},
                            Predicate{category_column, Comparison::equal, category},
                            Predicate{updated_column, Comparison::greater_equal, cutoff}};
using Query = std::array<float, dimension>;
using Queries = std::array<Query, query_count>;
using Top = std::array<ScoredRecord, top_k>;

void require(bool condition, const char* diagnostic) {
    if (!condition) { throw std::runtime_error(diagnostic); }
}

// Explicit SplitMix64 transition, unsigned wraparound and rejection-bounded shuffle.
// No standard-library distribution/shuffle implementation affects generated values.
class Random {
public:
    explicit Random(std::uint64_t initial) : state_(initial) {}
    std::uint64_t next() {
        auto value = (state_ += 0x9E3779B97F4A7C15ULL);
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31);
    }
    std::size_t index(std::size_t bound) {
        const auto wide = static_cast<std::uint64_t>(bound);
        const auto threshold = (std::uint64_t{0} - wide) % wide;
        std::uint64_t value;
        do { value = next(); } while (value < threshold);
        return static_cast<std::size_t>(value % wide);
    }
    float grid() { return static_cast<float>(static_cast<std::int32_t>(next() % 2049) - 1024) / 1024.0F; }
private:
    std::uint64_t state_;
};

// Application-owned oracle metadata, deliberately separate from ColumnBatch access.
struct Metadata {
    RecordID id;
    std::int64_t updated_at;
    std::int32_t tenant_id, category_id, token_count;
};
struct LogicalData {
    std::vector<Metadata> rows;
    std::vector<float> embeddings;
};

void preflight(const Cell& cell) {
    const auto embedding = cell.rows * dimension * sizeof(float);
    const auto numeric_ids = cell.rows * (3 * sizeof(std::int32_t) + sizeof(std::int64_t) + sizeof(RecordID));
    const auto oracle_metadata = cell.rows * sizeof(Metadata);
    const auto snapshot = embedding + numeric_ids;
    const auto oracle_candidates = cell.matches * sizeof(ScoredRecord);
    // ID-set allowance is a conservative policy estimate, NOT an allocator bound.
    const auto id_validation_allowance = cell.rows * 128;
    // Full-sort vector growth may hold old and new allocations at once.
    const auto reference_growth_allowance = 3 * oracle_candidates;
    const auto oracle_phase = embedding + oracle_metadata + oracle_candidates + cell.rows;
    const auto admission_phase = embedding + oracle_metadata + numeric_ids + snapshot + id_validation_allowance;
    const auto query_phase = snapshot + reference_growth_allowance;
    const auto runtime_allowance = std::size_t{64} * 1024 * 1024;
    const auto estimate = std::max({oracle_phase, admission_phase, query_phase}) + runtime_allowance;
    require(estimate <= memory_budget, "preflight estimate exceeds the 512 MiB budget");
    std::cout << "preflight cell=" << cell.name << " embedding_bytes=" << embedding
              << " numeric_id_bytes=" << numeric_ids << " oracle_metadata_bytes=" << oracle_metadata
              << " snapshot_payload_bytes=" << snapshot << " oracle_candidates_bytes=" << oracle_candidates
              << " id_validation_allowance_bytes=" << id_validation_allowance
              << " full_sort_growth_allowance_bytes=" << reference_growth_allowance
              << " estimated_bytes=" << estimate << " budget_bytes=" << memory_budget
              << " estimate_is_not_peak_RSS=true\n" << std::flush;
}

LogicalData generate(const Cell& cell) {
    Random embeddings(seed), selection(seed + 1);
    std::vector<unsigned char> mask(cell.rows, 0);
    std::fill_n(mask.begin(), cell.matches, static_cast<unsigned char>(1));
    for (std::size_t remaining = cell.rows; remaining > 1; --remaining) {
        std::swap(mask[remaining - 1], mask[selection.index(remaining)]);
    }
    LogicalData data{std::vector<Metadata>(cell.rows), std::vector<float>(cell.rows * dimension)};
    std::size_t failed = 0;
    for (std::size_t row = 0; row < cell.rows; ++row) {
        auto& metadata = data.rows[row];
        metadata = {RecordID{1'000'003} + RecordID{2} * static_cast<RecordID>(row),
                    cutoff + 1 + static_cast<std::int64_t>(row % 1000), tenant, category,
                    1 + static_cast<std::int32_t>(row % 2048)};
        if (mask[row] == 0) {
            switch (failed++ % 3) {
            case 0: metadata.tenant_id = 9; break;
            case 1: metadata.category_id = 3; break;
            case 2: metadata.updated_at = cutoff - 1 - static_cast<std::int64_t>(row % 1000); break;
            }
        }
        for (std::size_t feature = 0; feature < dimension; ++feature) {
            data.embeddings[row * dimension + feature] = embeddings.grid();
        }
    }
    return data;
}

Queries query_inputs() {
    Random random(seed + 2);
    Queries queries;
    for (std::size_t i = 0; i < queries.size(); ++i) {
        for (auto& value : queries[i]) { value = random.grid(); }
        for (std::size_t previous = 0; previous < i; ++previous) {
            require(queries[i] != queries[previous], "query vectors must be distinct");
        }
    }
    return queries;
}

struct Fingerprint {
    std::uint64_t value = 14'695'981'039'346'656'037ULL;
    void add(std::uint64_t word) {
        // Canonical little-endian words, independent of structure padding/host layout.
        for (unsigned byte = 0; byte < 8; ++byte) {
            value = (value ^ ((word >> (byte * 8)) & 255U)) * 1'099'511'628'211ULL;
        }
    }
};

NumericConstant field(const Metadata& row, std::size_t column) {
    switch (column) {
    case tenant_column: return row.tenant_id;
    case category_column: return row.category_id;
    case updated_column: return row.updated_at;
    case tokens_column: return row.token_count;
    default: throw std::out_of_range("oracle metadata column");
    }
}
template <typename T> bool compare(T value, T constant, Comparison comparison) {
    switch (comparison) {
    case Comparison::less: return value < constant;
    case Comparison::less_equal: return value <= constant;
    case Comparison::greater: return value > constant;
    case Comparison::greater_equal: return value >= constant;
    case Comparison::equal: return value == constant;
    case Comparison::not_equal: return value != constant;
    }
    throw std::invalid_argument("oracle comparison");
}
bool matches(const Metadata& row) {
    for (const auto& predicate : predicates) {
        const auto value = field(row, predicate.column_index);
        const auto match = std::visit([&]<typename T>(T constant) {
            const auto* typed = std::get_if<T>(&value);
            if (!typed) { throw std::invalid_argument("oracle predicate type"); }
            return compare(*typed, constant, predicate.comparison);
        }, predicate.constant);
        if (!match) { return false; }
    }
    return true;
}

struct Expected {
    std::size_t count = 0;
    std::int64_t tokens = 0;
    std::array<Top, query_count> ranked;
};
Expected independent_oracle(const LogicalData& data, const Queries& queries, const Cell& cell) {
    Expected expected;
    for (const auto& row : data.rows) {
        if (!matches(row)) { continue; }
        ++expected.count;
        const auto delta = static_cast<std::int64_t>(row.token_count);
        if ((delta > 0 && expected.tokens > std::numeric_limits<std::int64_t>::max() - delta) ||
            (delta < 0 && expected.tokens < std::numeric_limits<std::int64_t>::min() - delta)) {
            throw std::overflow_error("oracle token prefix overflow");
        }
        expected.tokens += delta;
    }
    require(expected.count == cell.matches, "actual candidate count differs from cell specification");
    for (std::size_t q = 0; q < queries.size(); ++q) {
        std::vector<ScoredRecord> candidates;
        candidates.reserve(expected.count);
        for (std::size_t row = 0; row < data.rows.size(); ++row) {
            if (!matches(data.rows[row])) { continue; }
            float score = 0.0F;
            for (std::size_t feature = 0; feature < dimension; ++feature) {
                score = std::fma(queries[q][feature], data.embeddings[row * dimension + feature], score);
                if (!std::isfinite(score)) { throw std::overflow_error("oracle nonfinite FMA prefix"); }
            }
            candidates.push_back({data.rows[row].id, score});
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            return a.score > b.score || (a.score == b.score && a.id < b.id);
        });
        require(candidates.size() == expected.count && candidates.size() >= top_k, "oracle candidate enumeration");
        std::copy_n(candidates.begin(), top_k, expected.ranked[q].begin());
        // Full sort for every query; only twenty owning results persist, not M-capacity vectors.
    }
    return expected;
}

void exact(std::span<const ScoredRecord> actual, std::span<const ScoredRecord> expected) {
    require(actual.size() == expected.size(), "Top-K result count mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(actual[i].id == expected[i].id, "Top-K RecordID/order mismatch");
        require(std::bit_cast<std::uint32_t>(actual[i].score) == std::bit_cast<std::uint32_t>(expected[i].score),
                "Top-K score bit mismatch");
    }
}

struct Inputs {
    ColumnBatch columns;
    EmbeddingBuffer embeddings;
    std::vector<RecordID> ids;
};
Inputs own_inputs(LogicalData&& logical) {
    const auto rows = logical.rows.size();
    std::vector<std::int32_t> tenants, categories, tokens;
    std::vector<std::int64_t> timestamps;
    std::vector<RecordID> ids;
    tenants.reserve(rows); categories.reserve(rows); tokens.reserve(rows); timestamps.reserve(rows); ids.reserve(rows);
    for (const auto& row : logical.rows) {
        tenants.push_back(row.tenant_id); categories.push_back(row.category_id);
        timestamps.push_back(row.updated_at); tokens.push_back(row.token_count); ids.push_back(row.id);
    }
    std::vector<ColumnBatch::Column> columns;
    columns.emplace_back(std::move(tenants)); columns.emplace_back(std::move(categories));
    columns.emplace_back(std::move(timestamps)); columns.emplace_back(std::move(tokens));
    return {ColumnBatch(std::move(columns)), EmbeddingBuffer(rows, dimension, std::move(logical.embeddings)), std::move(ids)};
}
Inputs construct_inputs(const Cell& cell) { return own_inputs(generate(cell)); }

std::uint64_t query_fingerprint(const Queries& queries) {
    Fingerprint hash;
    for (const auto& query : queries) { for (const auto value : query) { hash.add(std::bit_cast<std::uint32_t>(value)); } }
    return hash.value;
}
std::uint64_t result_fingerprint(std::span<const ScoredRecord> results) {
    Fingerprint hash;
    hash.add(results.size());
    for (const auto& value : results) { hash.add(value.id); hash.add(std::bit_cast<std::uint32_t>(value.score)); }
    return hash.value;
}
std::uint64_t input_fingerprint(const Inputs& inputs) {
    Fingerprint hash;
    const auto tenants = inputs.columns.column<std::int32_t>(tenant_column);
    const auto categories = inputs.columns.column<std::int32_t>(category_column);
    const auto timestamps = inputs.columns.column<std::int64_t>(updated_column);
    const auto tokens = inputs.columns.column<std::int32_t>(tokens_column);
    for (std::size_t row = 0; row < inputs.ids.size(); ++row) {
        hash.add(inputs.ids[row]); hash.add(static_cast<std::uint32_t>(tenants[row]));
        hash.add(static_cast<std::uint32_t>(categories[row])); hash.add(static_cast<std::uint64_t>(timestamps[row]));
        hash.add(static_cast<std::uint32_t>(tokens[row]));
    }
    for (std::size_t row = 0; row < inputs.ids.size(); ++row) {
        for (const auto value : inputs.embeddings.row(row)) { hash.add(std::bit_cast<std::uint32_t>(value)); }
    }
    return hash.value;
}

struct Golden {
    Expected expected;
    std::uint64_t data_hash = 0, query_hash = 0;
};
void export_golden(const std::string& path, const Cell& cell, const Golden& golden) {
    require(!std::filesystem::exists(path), "refusing to overwrite correctness expectations");
    std::ofstream output(path);
    require(output.good(), "cannot create correctness expectations");
    output << "VF_EXPECTATIONS_1 " << cell.name << ' ' << cell.rows << ' ' << golden.expected.count << ' '
           << golden.expected.tokens << ' ' << golden.data_hash << ' ' << golden.query_hash << '\n';
    for (std::size_t q = 0; q < query_count; ++q) {
        output << q << ' ' << result_fingerprint(golden.expected.ranked[q]) << '\n';
        for (const auto& value : golden.expected.ranked[q]) {
            output << value.id << ' ' << std::bit_cast<std::uint32_t>(value.score) << '\n';
        }
    }
    output.close(); require(output.good(), "cannot finish correctness expectations");
}
Golden read_golden(const std::string& path, const Cell& cell) {
    std::ifstream input(path);
    std::string version, name;
    std::size_t rows = 0;
    Golden golden;
    input >> version >> name >> rows >> golden.expected.count >> golden.expected.tokens >> golden.data_hash >> golden.query_hash;
    require(input.good() && version == "VF_EXPECTATIONS_1" && name == cell.name && rows == cell.rows &&
            golden.expected.count == cell.matches, "invalid correctness expectations header");
    for (std::size_t q = 0; q < query_count; ++q) {
        std::size_t index = query_count; std::uint64_t hash = 0;
        input >> index >> hash;
        require(input.good() && index == q, "invalid expected query ordering");
        for (auto& value : golden.expected.ranked[q]) {
            std::uint32_t bits = 0; input >> value.id >> bits; value.score = std::bit_cast<float>(bits);
            require(input.good() && std::isfinite(value.score), "invalid expected score");
        }
        require(hash == result_fingerprint(golden.expected.ranked[q]), "corrupt expected result fingerprint");
    }
    input >> std::ws; require(input.eof(), "unexpected trailing expectation data");
    return golden;
}

class NearestEnvironment {
public:
    NearestEnvironment() {
        require(std::fegetenv(&previous_) == 0, "cannot save floating-point environment");
        if (std::fesetround(FE_TONEAREST) != 0) {
            (void)std::fesetenv(&previous_);
            throw std::runtime_error("cannot set FE_TONEAREST");
        }
    }
    ~NearestEnvironment() { (void)std::fesetenv(&previous_); }
    NearestEnvironment(const NearestEnvironment&) = delete;
    NearestEnvironment& operator=(const NearestEnvironment&) = delete;
private:
    std::fenv_t previous_;
};

void run(const Cell& cell, const std::string& export_path = {}) {
    preflight(cell);
    const NearestEnvironment environment;
    const auto queries = query_inputs();
    Fingerprint query_hash, data_hash;
    for (const auto& query : queries) { for (const auto value : query) { query_hash.add(std::bit_cast<std::uint32_t>(value)); } }
    Expected expected;
    std::optional<ValidatedDataset> snapshot;
    {
        auto logical = generate(cell);
        expected = independent_oracle(logical, queries, cell); // Never calls a VectorForge executor/reference.
        for (const auto& row : logical.rows) {
            data_hash.add(row.id); data_hash.add(static_cast<std::uint32_t>(row.tenant_id));
            data_hash.add(static_cast<std::uint32_t>(row.category_id)); data_hash.add(static_cast<std::uint64_t>(row.updated_at));
            data_hash.add(static_cast<std::uint32_t>(row.token_count));
        }
        for (const auto value : logical.embeddings) { data_hash.add(std::bit_cast<std::uint32_t>(value)); }
        const auto inputs = own_inputs(std::move(logical));
        snapshot.emplace(inputs.columns, inputs.embeddings, inputs.ids);
        require(snapshot->columns().row_count() == cell.rows && snapshot->embeddings().row_count() == cell.rows,
                "snapshot row alignment");
        require(snapshot->columns().column<std::int32_t>(tenant_column).data() != inputs.columns.column<std::int32_t>(tenant_column).data()
                && snapshot->embeddings().row(0).data() != inputs.embeddings.row(0).data()
                && snapshot->record_ids().data() != inputs.ids.data(), "snapshot ownership must be independent");
        for (std::size_t row = 0; row < cell.rows; ++row) {
            require(snapshot->record_ids()[row] == logical.rows[row].id, "snapshot ID mapping");
        }
    } // All original inputs and oracle-owned logical data are destroyed before any query.

    const auto count = count_where(*snapshot, predicates, batch_size);
    // Existing raw-column SUM, even though these columns belong to an admitted snapshot.
    const auto tokens = sum_where(snapshot->columns(), tokens_column, predicates, batch_size);
    require(count == expected.count && tokens == expected.tokens, "candidate COUNT/SUM mismatch");
    std::vector<ScoredRecord> retained;
    for (std::size_t q = 0; q < query_count; ++q) {
        auto bounded = execute_batched(*snapshot, queries[q], predicates, top_k, batch_size);
        const auto full_sort = execute_scalar_reference(*snapshot, queries[q], predicates, top_k);
        exact(bounded, expected.ranked[q]);
        exact(full_sort, expected.ranked[q]);
        exact(bounded, full_sort);
        if (q == 0) { retained = std::move(bounded); }
        std::cout << "cell=" << cell.name << " query=" << q << " exact_top_k=" << top_k << " three_way_agreement=true\n";
    }
    snapshot.reset();
    exact(retained, expected.ranked[0]);
    std::cout << "PASS cell=" << cell.name << " rows=" << cell.rows << " candidates=" << count << " token_sum=" << tokens
              << " queries=" << query_count << " top_k=" << top_k << " batch=" << batch_size
              << " tail_rows=" << cell.rows % batch_size << " seed=" << seed
              << " data_fingerprint=" << data_hash.value << " query_fingerprint=" << query_hash.value
              << " source_released_before_queries=true owned_result_survives_snapshot=true\n";
    if (!export_path.empty()) { export_golden(export_path, cell, Golden{expected, data_hash.value, query_hash.value}); }
    // The default path remains the complete, untimed correctness gate.
    // Ordinary finite grid inputs do not establish arbitrary libm-provider side-effect equivalence.
}

using Tick = std::int64_t;
Tick now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct QueryObservation { Tick start = 0, end = 0; std::size_t capacity_bytes = 0; };
struct Session {
    Tick start = 0, end = 0;
    std::size_t count = 0;
    std::int64_t tokens = 0;
    std::array<Top, query_count> sink{};
    std::array<QueryObservation, query_count> query{};
};
struct Measurement {
    const Cell& cell;
    const Golden& golden;
    std::string_view mode, path, run_id;
};

void session(const ValidatedDataset& snapshot, const Queries& queries, bool bounded, Session& observation, bool timed) {
    if (timed) { observation.start = now_ns(); }
    observation.count = count_where(snapshot, predicates, batch_size);
    observation.tokens = sum_where(snapshot.columns(), tokens_column, predicates, batch_size);
    for (std::size_t q = 0; q < query_count; ++q) {
        if (timed) { observation.query[q].start = now_ns(); }
        {
            const auto result = bounded ? execute_batched(snapshot, queries[q], predicates, top_k, batch_size)
                                        : execute_scalar_reference(snapshot, queries[q], predicates, top_k);
            if (timed) { observation.query[q].end = now_ns(); }
            require(result.size() == top_k, "measurement result size changed");
            observation.query[q].capacity_bytes = result.capacity() * sizeof(ScoredRecord);
            std::copy(result.begin(), result.end(), observation.sink[q].begin());
        } // Result destruction and sink copy are inside the session, outside query-call timing.
    }
    if (timed) { observation.end = now_ns(); }
}
std::uint64_t session_fingerprint(const Session& observation) {
    Fingerprint hash; hash.add(observation.count); hash.add(static_cast<std::uint64_t>(observation.tokens));
    for (const auto& result : observation.sink) { hash.add(result_fingerprint(result)); }
    return hash.value;
}
void validate(const Session& observation, const Golden& golden) {
    require(observation.count == golden.expected.count && observation.tokens == golden.expected.tokens,
            "measurement COUNT/SUM differs from independent expectations");
    for (std::size_t q = 0; q < query_count; ++q) { exact(observation.sink[q], golden.expected.ranked[q]); }
}
void common(const Measurement& m, std::string_view record, std::uint64_t result_hash) {
    std::cout << "{\"record\":\"" << record << "\",\"mode\":\"" << m.mode << "\",\"cell\":\"" << m.cell.name
              << "\",\"path\":\"" << m.path << "\",\"run_id\":\"" << m.run_id << "\",\"n\":" << m.cell.rows
              << ",\"m\":" << m.cell.matches << ",\"d\":" << dimension << ",\"k\":" << top_k << ",\"batch\":" << batch_size
              << ",\"scored_record_bytes\":" << sizeof(ScoredRecord) << ",\"count\":" << m.golden.expected.count
              << ",\"token_sum\":" << m.golden.expected.tokens << ",\"data_fp\":" << m.golden.data_hash
              << ",\"query_fp\":" << m.golden.query_hash << ",\"result_fp\":" << result_hash;
}
void interval(const Measurement& m, std::string_view record, std::string_view stage, Tick start, Tick end, std::uint64_t hash) {
    require(end >= start, "non-monotonic measurement interval");
    common(m, record, hash);
    std::cout << ",\"stage\":\"" << stage << "\",\"start_ns\":" << start << ",\"end_ns\":" << end
              << ",\"duration_ns\":" << end - start << "}\n";
}
void emit_session(const Measurement& m, const Session& observation, std::string_view period, std::size_t index) {
    common(m, "session", session_fingerprint(observation));
    std::cout << ",\"period\":\"" << period << "\",\"session\":" << index << ",\"start_ns\":" << observation.start
              << ",\"end_ns\":" << observation.end << ",\"duration_ns\":" << observation.end - observation.start << "}\n";
    for (std::size_t q = 0; q < query_count; ++q) {
        common(m, "query", result_fingerprint(observation.sink[q]));
        std::cout << ",\"period\":\"" << period << "\",\"session\":" << index << ",\"query\":" << q
                  << ",\"start_ns\":" << observation.query[q].start << ",\"end_ns\":" << observation.query[q].end
                  << ",\"duration_ns\":" << observation.query[q].end - observation.query[q].start
                  << ",\"result_capacity_bytes\":" << observation.query[q].capacity_bytes << "}\n";
    }
}

void timing(const Measurement& m, const Queries& queries) {
    // All logs/results have fixed small storage allocated BEFORE the continuous B..session10 window.
    std::array<Session, 10> initial{}, steady{};
    std::array<Session, 2> warmup{};
    std::optional<Inputs> source;
    std::optional<ValidatedDataset> snapshot;
    const auto a0 = now_ns(); source.emplace(construct_inputs(m.cell)); const auto a1 = now_ns();
    // Input binding scans are outside construction and continuous admission/session timing.
    require(input_fingerprint(*source) == m.golden.data_hash, "measurement dataset fingerprint changed");
    const auto b0 = now_ns(); snapshot.emplace(source->columns, source->embeddings, source->ids); const auto b1 = now_ns();
    const auto c0 = now_ns(); source.reset(); const auto c1 = now_ns();
    for (auto& observation : initial) { session(*snapshot, queries, m.path == "bounded", observation, true); }
    // No oracle evaluation, output, fingerprinting or expected-result comparisons above.
    for (const auto& observation : initial) { validate(observation, m.golden); }
    for (auto& observation : warmup) {
        session(*snapshot, queries, m.path == "bounded", observation, false); validate(observation, m.golden);
    }
    for (auto& observation : steady) {
        session(*snapshot, queries, m.path == "bounded", observation, true); validate(observation, m.golden);
    }
    const auto e0 = now_ns(); snapshot.reset(); const auto e1 = now_ns();
    const auto hash = session_fingerprint(initial.front());
    interval(m, "phase", "construction", a0, a1, hash);
    interval(m, "phase", "admission", b0, b1, hash);
    interval(m, "phase", "source_release", c0, c1, hash);
    for (std::size_t i = 0; i < initial.size(); ++i) {
        emit_session(m, initial[i], "initial", i);
        if (i == 0 || i == 9) { interval(m, "prefix", i == 0 ? "one_session" : "ten_sessions", b0, initial[i].end, hash); }
    }
    common(m, "warmup", hash); std::cout << ",\"sessions\":2}\n";
    for (std::size_t i = 0; i < steady.size(); ++i) { emit_session(m, steady[i], "steady", i); }
    interval(m, "phase", "snapshot_release", e0, e1, hash);
    common(m, "complete", hash); std::cout << ",\"sessions\":22,\"steady_sessions\":10,\"queries\":176}\n";
}

void checkpoint(const Measurement& m, std::string_view stage, std::uint64_t hash) {
    std::ifstream status("/proc/self/status");
    require(status.good(), "memory mode requires Linux /proc/self/status");
    std::string line;
    std::uint64_t rss = 0, hwm = 0;
    while (std::getline(status, line)) {
        if (line.starts_with("VmRSS:")) { rss = std::stoull(line.substr(6)) * 1024; }
        if (line.starts_with("VmHWM:")) { hwm = std::stoull(line.substr(6)) * 1024; }
    }
    require(rss > 0 && hwm > 0, "missing process RSS counters");
    common(m, "checkpoint", hash);
    std::cout << ",\"stage\":\"" << stage << "\",\"timestamp_ns\":" << now_ns()
              << ",\"rss_bytes\":" << rss << ",\"os_hwm_bytes\":" << hwm << "}\n" << std::flush;
}
void memory(const Measurement& m, const Queries& queries) {
    Session observation{};
    // Small expected-output storage is included in the baseline; no oracle dataset or scores exist here.
    observation.count = m.golden.expected.count; observation.tokens = m.golden.expected.tokens;
    observation.sink = m.golden.expected.ranked;
    const auto hash = session_fingerprint(observation);
    checkpoint(m, "before_construction", hash);
    std::optional<Inputs> source; source.emplace(construct_inputs(m.cell));
    checkpoint(m, "after_construction", hash);
    require(input_fingerprint(*source) == m.golden.data_hash, "memory dataset fingerprint changed");
    std::optional<ValidatedDataset> snapshot; snapshot.emplace(source->columns, source->embeddings, source->ids);
    checkpoint(m, "after_admission", hash);
    source.reset(); checkpoint(m, "after_source_release", hash);
    for (std::size_t i = 0; i < 10; ++i) {
        session(*snapshot, queries, m.path == "bounded", observation, false); validate(observation, m.golden);
    }
    checkpoint(m, "after_sessions", session_fingerprint(observation));
    snapshot.reset(); checkpoint(m, "after_snapshot_release", hash);
    common(m, "complete", hash);
    std::cout << ",\"sessions\":10,\"queries\":80,\"allocation_calls\":null,\"allocation_scope\":\"unavailable_without_global_allocator_override\",\"result_capacities_bytes\":[";
    for (std::size_t q = 0; q < query_count; ++q) { if (q != 0) { std::cout << ','; } std::cout << observation.query[q].capacity_bytes; }
    std::cout << "]}\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 2, "usage: workload CELL [--export FILE | --mode timing|memory --path bounded|full-sort --run-id ID --expectations FILE]");
        const auto cell = std::find_if(cells.begin(), cells.end(), [&](const auto& value) { return value.name == argv[1]; });
        if (cell == cells.end()) { throw std::invalid_argument("unknown workload cell"); }
        if (argc == 2) { run(*cell); return 0; }
        if (argc == 4 && std::string_view(argv[2]) == "--export") { run(*cell, argv[3]); return 0; }
        require(argc == 10 && std::string_view(argv[2]) == "--mode" && std::string_view(argv[4]) == "--path" &&
                std::string_view(argv[6]) == "--run-id" && std::string_view(argv[8]) == "--expectations", "invalid measurement arguments");
        const std::string_view mode = argv[3], path = argv[5], run_id = argv[7];
        require((mode == "timing" || mode == "memory") && (path == "bounded" || path == "full-sort"), "invalid measurement mode/path");
        require(!run_id.empty() && run_id.size() <= 80 && std::all_of(run_id.begin(), run_id.end(), [](char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
        }), "invalid run ID");
        const auto golden = read_golden(argv[9], *cell);
        const auto queries = query_inputs();
        require(query_fingerprint(queries) == golden.query_hash, "measurement query fingerprint changed");
        const NearestEnvironment environment;
        const Measurement measurement{*cell, golden, mode, path, run_id};
        if (mode == "timing") { timing(measurement, queries); } else { memory(measurement, queries); }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Workload exactness gate failed: " << error.what() << '\n'; return 1;
    }
}
