#include "scoring.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <charconv>

// Compile the historical fixture and oracle in this standalone executable only.
// Its benchmark/CLI entry point is renamed and is never used by the experiment.
#define main historical_workload_entry
#include "../../integration/ai_retrieval/workload.cpp"
#undef main

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace {
namespace gpu = cuda_scoring_pilot;

struct ScanPredicate {
    ColumnBatch::ColumnView column;
    NumericConstant constant;
    Comparison op;
};

template<class T>
bool row_compare(T lhs, T rhs, Comparison op) {
    switch (op) {
        case Comparison::less: return lhs < rhs;
        case Comparison::less_equal: return lhs <= rhs;
        case Comparison::greater: return lhs > rhs;
        case Comparison::greater_equal: return lhs >= rhs;
        case Comparison::equal: return lhs == rhs;
        case Comparison::not_equal: return lhs != rhs;
    }
    throw std::invalid_argument("unsupported comparison");
}

void validate_operator(Comparison op) {
    switch (op) {
        case Comparison::less:
        case Comparison::less_equal:
        case Comparison::greater:
        case Comparison::greater_equal:
        case Comparison::equal:
        case Comparison::not_equal: return;
    }
    throw std::invalid_argument("unsupported comparison");
}

bool accepts(const ScanPredicate& predicate, std::size_t row) {
    return std::visit([&]<class T>(T constant) {
        return row_compare(std::get<std::span<const T>>(predicate.column)[row], constant, predicate.op);
    }, predicate.constant);
}

struct Diagnostic {
    Tick validate = 0, row_reserve = 0, scan_emit = 0, rank = 0;
    gpu::ScorePhases scoring;
};
struct CandidateTrace {
    std::vector<gpu::RowIndex> rows;
    std::vector<gpu::Score> scores;
};
#ifdef VECTORFORGE_V2_MIXED
struct MixedObservation {
    std::size_t matched = 0;
    Tick verify_ns = 0, map_ns = 0, topk_ns = 0;
    gpu::DevicePhases* events = nullptr;
};
#endif
struct ScoreFailure : std::overflow_error {
    ScoreFailure(std::size_t slot, gpu::RowIndex row, std::uint32_t feature)
        : std::overflow_error("GPU nonfinite score at slot " + std::to_string(slot)
                              + ", row " + std::to_string(row) + ", dimension " + std::to_string(feature)),
          slot(slot), row(row), feature(feature) {}
    std::size_t slot;
    gpu::RowIndex row;
    std::uint32_t feature;
};

// Contract G, only over the provided admitted snapshot. Resolve typed views and
// validate every operand even for K=0. Scan predicates in order and stop on false.
// Materialize candidate and score arrays. CPU partial_sort uses O(M) records and
// the same score/ID ordering as production, with no result published on failure.
std::vector<ScoredRecord> retrieve_gpu(const ValidatedDataset& snapshot, gpu::ResidentScorer& scorer,
                                      std::span<const float> q, std::span<const Predicate> filters,
                                      std::size_t k, std::size_t batch, Diagnostic* phases = nullptr,
                                      CandidateTrace* trace = nullptr
#ifdef VECTORFORGE_V2_MIXED
                                      , MixedObservation* mixed = nullptr
#endif
                                      ) {
    const auto begin = phases ? now_ns() : 0;
    if (batch == 0) { throw std::invalid_argument("batch size must be positive"); }
    const auto& cols = snapshot.columns();
    const auto& emb = snapshot.embeddings();
    if (q.size() != emb.dimension() || !std::ranges::all_of(q, [](float x) { return std::isfinite(x); })) {
        throw std::invalid_argument("query must match the snapshot dimension and be finite");
    }
    if (cols.row_count() > gpu::no_failure) { throw std::length_error("candidate index range"); }
    std::vector<ScanPredicate> bound;
    bound.reserve(filters.size());
    for (const auto& predicate : filters) {
        validate_operator(predicate.comparison);
        const auto& literal = predicate.constant;
        const auto view = std::visit([&]<class T>(T value) -> ColumnBatch::ColumnView {
            const auto column = cols.column<T>(predicate.column_index);
            if constexpr (std::same_as<T, float>) {
                if (!std::isfinite(value)) { throw std::invalid_argument("non-finite predicate"); }
            }
            return column;
        }, literal);
        bound.push_back({view, literal, predicate.comparison});
    }
    const auto ids = snapshot.record_ids();
    const auto valid = phases ? now_ns() : 0;
    if (phases) { phases->validate = valid - begin; }
    if (k == 0) { return {}; }

    std::vector<gpu::RowIndex> rows;
    rows.reserve(cols.row_count());
    const auto reserved = phases ? now_ns() : 0;
    for (std::size_t offset = 0; offset < cols.row_count();) {
        const auto end = offset + std::min(batch, cols.row_count() - offset);
        for (std::size_t row = offset; row < end; ++row) {
            if (std::ranges::all_of(bound, [row](const ScanPredicate& p) { return accepts(p, row); })) {
                rows.push_back(static_cast<gpu::RowIndex>(row));
            }
        }
        offset = end;
    }
    if (phases) {
        phases->row_reserve = reserved - valid;
        phases->scan_emit = now_ns() - reserved;
    }
#ifdef VECTORFORGE_V2_MIXED
    if (mixed) { mixed->matched = rows.size(); }
    auto scores = scorer.score(rows, q, phases ? &phases->scoring : nullptr, mixed ? mixed->events : nullptr);
#else
    auto scores = scorer.score(rows, q, phases ? &phases->scoring : nullptr);
#endif
    const auto rank_start = phases ? now_ns() : 0;
    require(scores.size() == rows.size(), "GPU score count changed");
    for (std::size_t slot = 0; slot < scores.size(); ++slot) {
        if (scores[slot].first_failure != gpu::no_failure) {
            throw ScoreFailure(slot, rows[slot], scores[slot].first_failure);
        }
        require(std::isfinite(scores[slot].value), "non-finite successful GPU score");
    }
#ifdef VECTORFORGE_V2_MIXED
    const auto verified = phases && mixed ? now_ns() : 0;
#endif
    std::vector<ScoredRecord> records;
    records.reserve(rows.size());
    for (std::size_t slot = 0; slot < rows.size(); ++slot) {
        records.push_back({ids[rows[slot]], scores[slot].value});
    }
#ifdef VECTORFORGE_V2_MIXED
    const auto mapped = phases && mixed ? now_ns() : 0;
#endif
    const auto limit = std::min(k, records.size());
    std::partial_sort(records.begin(), records.begin() + static_cast<std::ptrdiff_t>(limit), records.end(),
                      [](const auto& a, const auto& b) {
                          return a.score > b.score || (a.score == b.score && a.id < b.id);
                      });
    records.resize(limit);
    if (phases) {
        const auto finished = now_ns();
        phases->rank = finished - rank_start;
#ifdef VECTORFORGE_V2_MIXED
        if (mixed) {
            mixed->verify_ns = verified - rank_start;
            mixed->map_ns = mapped - verified;
            mixed->topk_ns = finished - mapped;
        }
#endif
    }
    if (trace) { *trace = {std::move(rows), std::move(scores)}; }
    return records;
}

std::vector<float> pack(const ValidatedDataset& snapshot) {
    const auto& values = snapshot.embeddings();
    // EmbeddingBuffer admission already checks the total element count.
    std::vector<float> flat;
    flat.reserve(values.row_count() * values.dimension());
    for (std::size_t row = 0; row < values.row_count(); ++row) {
        const auto view = values.row(row);
        flat.insert(flat.end(), view.begin(), view.end());
    }
    return flat;
}

void create_resident(std::optional<gpu::ResidentScorer>& scorer, const ValidatedDataset& snapshot,
                     const std::vector<float>& flat, gpu::ResidencyStats& stats) {
    scorer.emplace(flat, snapshot.embeddings().row_count(), snapshot.embeddings().dimension(),
                   snapshot.columns().row_count(), false, &stats);
}

gpu::Score oracle_score(std::span<const float> embedding, std::span<const float> q) {
    float score = +0.0f;
    for (std::size_t d = 0; d < embedding.size(); ++d) {
        score = std::fma(q[d], embedding[d], score);
        if (!std::isfinite(score)) { return {0.0f, static_cast<std::uint32_t>(d)}; }
    }
    return {score, gpu::no_failure};
}

void verify_candidate(CandidateTrace const& trace, std::size_t slot, gpu::RowIndex row, RecordID id,
                      std::span<const float> embedding, std::span<const float> q, std::ofstream& bits_file) {
    if (trace.rows[slot] != row) {
        std::cerr << "candidate mismatch slot=" << slot << " expected_row=" << row
                  << " actual_row=" << trace.rows[slot] << '\n';
        throw std::runtime_error("candidate order or identity differs from logical oracle");
    }
    const auto expected = oracle_score(embedding, q);
    const auto got = trace.scores[slot];
    const std::array words{row, static_cast<std::uint32_t>(id), static_cast<std::uint32_t>(id >> 32),
                           std::bit_cast<std::uint32_t>(expected.value), std::bit_cast<std::uint32_t>(got.value),
                           expected.first_failure, got.first_failure};
    static_assert(std::endian::native == std::endian::little);
    bits_file.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(sizeof(words)));
    if (words[3] != words[4] || words[5] != words[6]) {
        std::cerr << "score mismatch slot=" << slot << " row=" << row << " id=" << id << '\n';
        for (const auto values : {embedding, q}) {
            for (const float v : values) { std::cerr << ' ' << std::hex << std::bit_cast<std::uint32_t>(v); }
            std::cerr << std::dec << '\n';
        }
        throw std::runtime_error("candidate score bits/status differ from independent oracle");
    }
}

void correctness(const Cell& cell, const Golden& golden, const char* raw_path) {
    const auto queries = query_inputs();
    const auto logical = generate(cell);
    const auto expected = independent_oracle(logical, queries, cell);
    require(expected.count == golden.expected.count && expected.tokens == golden.expected.tokens,
            "oracle aggregates differ from historical evidence");
    std::optional<ValidatedDataset> snapshot;
    {
        const auto inputs = own_inputs(LogicalData{logical.rows, logical.embeddings});
        require(input_fingerprint(inputs) == golden.data_hash && query_fingerprint(queries) == golden.query_hash,
                "historical fingerprint changed");
        snapshot.emplace(inputs.columns, inputs.embeddings, inputs.ids);
    }
    require(count_where(*snapshot, predicates, batch_size) == golden.expected.count &&
                sum_where(snapshot->columns(), tokens_column, predicates, batch_size) == golden.expected.tokens,
            "admitted aggregate mismatch");

    gpu::select_device();
    gpu::ResidencyStats stats;
    std::optional<gpu::ResidentScorer> scorer;
    {
        const auto flat = pack(*snapshot);
        create_resident(scorer, *snapshot, flat, stats);
    }
    std::ofstream raw(raw_path, std::ios::binary | std::ios::trunc);
    require(raw.good(), "cannot create raw candidate comparison file");
    for (std::size_t j = 0; j < queries.size(); ++j) {
        const auto& query = queries[j];
        CandidateTrace trace;
        const auto top = retrieve_gpu(*snapshot, *scorer, query, predicates, top_k, batch_size, nullptr, &trace);
        require(trace.rows.size() == cell.matches && trace.scores.size() == cell.matches, "candidate cardinality");
        std::size_t slot = 0;
        for (std::size_t row = 0; row < logical.rows.size(); ++row) {
            if (!matches(logical.rows[row])) { continue; }
            require(slot < trace.rows.size() && snapshot->record_ids()[row] == logical.rows[row].id, "candidate ID mismatch");
            verify_candidate(trace, slot++, static_cast<gpu::RowIndex>(row), logical.rows[row].id,
                             std::span(logical.embeddings).subspan(row * dimension, dimension), query, raw);
        }
        require(slot == cell.matches, "independent enumeration differed");
        exact(top, golden.expected.ranked[j]);
        exact(top, expected.ranked[j]);
        exact(top, execute_batched(*snapshot, query, predicates, top_k, batch_size));
        exact(top, execute_scalar_reference(*snapshot, query, predicates, top_k));
        std::cout << "gate cell=" << cell.name << " query=" << j << " slots=" << slot
                  << " top_fp=" << result_fingerprint(top) << " exact=true\n";
    }
    raw.close();
    require(raw.good(), "cannot finish candidate comparison file");
    scorer->close();
    std::cout << "PASS gate " << cell.name << " data_fp=" << golden.data_hash << " query_fp=" << golden.query_hash
              << " count=" << expected.count << " sum=" << expected.tokens << " free_device_bytes=" << stats.free_device_bytes
              << " resident_bytes=" << stats.embedding_bytes << " device_bytes=" << stats.device_bytes
              << " pinned_bytes=" << stats.pinned_bytes << '\n';
}

struct TimedSequence {
    Tick total = 0;
    std::array<Tick, query_count> queries{};
    std::array<Diagnostic, query_count> phases{};
};

TimedSequence timed_sequence(const ValidatedDataset& snapshot, gpu::ResidentScorer* scorer,
                             const Queries& queries, const Golden& golden, bool diagnose) {
    TimedSequence t;
    std::array<Top, query_count> consumed{};
    const auto whole = now_ns();
    for (std::size_t j = 0; j < queries.size(); ++j) {
        const auto begin = now_ns();
        {
            const auto result = scorer ? retrieve_gpu(snapshot, *scorer, queries[j], predicates, top_k, batch_size,
                                                      diagnose ? &t.phases[j] : nullptr)
                                       : execute_batched(snapshot, queries[j], predicates, top_k, batch_size);
            require(result.size() == top_k, "timed result size changed");
            std::copy(result.begin(), result.end(), consumed[j].begin());
        }
        t.queries[j] = now_ns() - begin;
    }
    t.total = now_ns() - whole;
    for (std::size_t j = 0; j < queries.size(); ++j) { exact(consumed[j], golden.expected.ranked[j]); }
    return t;
}

void emit_sequence(const char* kind, std::size_t index, const TimedSequence& t, bool diagnose) {
    std::cout << "{\"kind\":\"" << kind << "\",\"index\":" << index << ",\"sequence_ns\":" << t.total << ",\"query_ns\":[";
    for (std::size_t j = 0; j < t.queries.size(); ++j) { std::cout << (j ? "," : "") << t.queries[j]; }
    std::cout << "]}\n";
    if (!diagnose) { return; }
    for (std::size_t j = 0; j < t.phases.size(); ++j) {
        const auto& p = t.phases[j];
        const auto& s = p.scoring;
        std::cout << "{\"kind\":\"phases\",\"sequence\":" << index << ",\"query\":" << j
                  << ",\"total_ns\":" << t.queries[j] << ",\"validation_ns\":" << p.validate
                  << ",\"row_reserve_ns\":" << p.row_reserve << ",\"filter_and_emit_ns\":" << p.scan_emit
                  << ",\"score_prepare_ns\":" << s.prepare_ns << ",\"h2d_submit_ns\":" << s.h2d_submit_ns
                  << ",\"h2d_wait_ns\":" << s.h2d_wait_ns << ",\"kernel_submit_ns\":" << s.kernel_submit_ns
                  << ",\"kernel_wait_ns\":" << s.kernel_wait_ns << ",\"d2h_submit_ns\":" << s.d2h_submit_ns
                  << ",\"d2h_wait_ns\":" << s.d2h_wait_ns << ",\"score_unpack_ns\":" << s.unpack_ns
                  << ",\"rank_ns\":" << p.rank << "}\n";
    }
}

void measure(const Cell& cell, const Golden& golden, bool device, bool diagnose) {
    const auto generate_start = now_ns();
    std::optional<Inputs> inputs(construct_inputs(cell));
    const auto queries = query_inputs();
    const auto generated = now_ns();
    require(input_fingerprint(*inputs) == golden.data_hash && query_fingerprint(queries) == golden.query_hash,
            "historical fingerprint changed");
    const auto admission_start = now_ns();
    const ValidatedDataset snapshot(inputs->columns, inputs->embeddings, inputs->ids);
    const auto admitted = now_ns();
    inputs.reset();
    const auto source_released = now_ns();
    std::optional<gpu::ResidentScorer> owned;
    gpu::ResidencyStats stats;
    Tick context = 0, pack_time = 0, residency = 0;
    if (device) {
        const auto context_start = now_ns();
        gpu::select_device(false);
        const auto context_end = now_ns();
        context = context_end - context_start;
        {
            const auto flat = pack(snapshot);
            const auto packed = now_ns();
            pack_time = packed - context_end;
            create_resident(owned, snapshot, flat, stats);
        }
        residency = now_ns() - context_end - pack_time;
    }
    auto* scorer = owned ? &*owned : nullptr;
    const auto first = timed_sequence(snapshot, scorer, queries, golden, false);
    for (int i = 0; i < 2; ++i) { (void)timed_sequence(snapshot, scorer, queries, golden, false); }
    std::array<TimedSequence, 8> samples{};
    const auto count = diagnose ? std::size_t{2} : samples.size();
    for (std::size_t i = 0; i < count; ++i) { samples[i] = timed_sequence(snapshot, scorer, queries, golden, diagnose); }
    const auto cleanup_start = now_ns();
    if (owned) { owned->close(); }
    const auto cleanup = now_ns() - cleanup_start;

    std::cout << "{\"kind\":\"setup\",\"cell\":\"" << cell.name << "\",\"path\":\"" << (device ? "gpu" : "cpu")
              << "\",\"data_fp\":" << golden.data_hash << ",\"query_fp\":" << golden.query_hash
              << ",\"construction_ns\":" << generated - generate_start << ",\"admission_ns\":" << admitted - admission_start
              << ",\"source_release_ns\":" << source_released - admitted << ",\"context_ns\":" << context
              << ",\"pack_ns\":" << pack_time << ",\"residency_ns\":" << residency << ",\"cleanup_ns\":" << (device ? cleanup : 0)
              << ",\"allocation_ns\":" << stats.allocation_ns << ",\"embedding_upload_ns\":" << stats.upload_ns
              << ",\"free_device_bytes\":" << stats.free_device_bytes << ",\"resident_bytes\":" << stats.embedding_bytes
              << ",\"device_bytes\":" << stats.device_bytes << ",\"pinned_bytes\":" << stats.pinned_bytes << "}\n";
    emit_sequence("cold", 0, first, false);
    for (std::size_t i = 0; i < count; ++i) { emit_sequence("steady", i, samples[i], diagnose); }
    std::cout << "{\"kind\":\"verified\",\"query_order\":[0,1,2,3,4,5,6,7],\"warmup_sequences\":2,\"exact\":true}\n";
}

template<class Exception, class Call>
void reject_input(const char* label, Call&& call) {
    try { call(); }
    catch (const Exception& error) {
        std::cout << "expected rejection " << label << " " << error.what() << '\n';
        return;
    }
    throw std::runtime_error("expected experimental consumer rejection");
}

void fixtures() {
    gpu::select_device();
    constexpr std::size_t total = 131;
    const float tiny = std::numeric_limits<float>::denorm_min();
    const auto snapshot = [&] {
        std::vector<std::int32_t> counters(total);
        std::vector<std::int64_t> stamps(total, 9007199254740993LL);
        std::vector<float> floats(total, tiny);
        std::vector<RecordID> ids(total);
        std::vector<float> embeddings(total * 3, 0.0f);
        for (std::size_t j = 0; j < total; ++j) {
            counters[j] = static_cast<std::int32_t>(j);
            ids[j] = total - j;
            embeddings[j * 3] = j % 2 == 0 ? 1.0f : 0x1.000002p0f;
        }
        embeddings[3 * (total - 1)] = -tiny;
        const ColumnBatch cols({std::move(counters), std::move(stamps), std::move(floats)});
        const EmbeddingBuffer emb(total, 3, std::move(embeddings));
        return ValidatedDataset(cols, emb, ids);
    }();
    gpu::ResidencyStats stats;
    std::optional<gpu::ResidentScorer> scorer;
    create_resident(scorer, snapshot, pack(snapshot), stats);
    const std::array q{0.5f, 1.0f, 1.0f};
    constexpr std::array counts{0, 1, 31, 32, 33, 127, 128, 129, 131};
    for (const auto count : counts) {
        const std::array filters{Predicate{0, Comparison::less, std::int32_t{count}},
                                 Predicate{1, Comparison::equal, std::int64_t{9007199254740993LL}},
                                 Predicate{2, Comparison::greater, 0.0f}};
        for (const auto k : {std::size_t{0}, std::size_t{1}, std::size_t{32}, total + 1}) {
            CandidateTrace trace;
            const auto got = retrieve_gpu(snapshot, *scorer, q, filters, k, 32, nullptr, &trace);
            exact(got, execute_batched(snapshot, q, filters, k, 32));
            exact(got, execute_scalar_reference(snapshot, q, filters, k));
            require(trace.rows.size() == (k ? static_cast<std::size_t>(count) : 0), "fixture candidates");
            for (std::size_t i = 0; i < trace.rows.size(); ++i) {
                require(trace.rows[i] == i, "fixture row order");
                const auto s = oracle_score(snapshot.embeddings().row(i), q);
                require(std::bit_cast<std::uint32_t>(s.value) == std::bit_cast<std::uint32_t>(trace.scores[i].value)
                        && s.first_failure == trace.scores[i].first_failure, "fixture candidate score");
            }
            std::cout << "boundary count=" << count << " k=" << k << " batch=32 exact=true\n";
        }
    }
    exact(retrieve_gpu(snapshot, *scorer, q, {}, total, 32), execute_scalar_reference(snapshot, q, {}, total));
    const std::array bad_column{Predicate{9, Comparison::equal, std::int32_t{0}}};
    const std::array bad_op{Predicate{9, static_cast<Comparison>(255), std::int32_t{0}}};
    const std::array mismatched{Predicate{1, Comparison::equal, 1.0f}};
    const std::array nonfinite{Predicate{2, Comparison::equal, std::numeric_limits<float>::infinity()}};
    const std::array bad_query{0.0f, 1.0f, std::numeric_limits<float>::quiet_NaN()};
    reject_input<std::out_of_range>("column", [&] { (void)retrieve_gpu(snapshot, *scorer, q, bad_column, 0, 32); });
    reject_input<std::invalid_argument>("operator before column", [&] { (void)retrieve_gpu(snapshot, *scorer, q, bad_op, 0, 32); });
    reject_input<std::invalid_argument>("predicate type", [&] { (void)retrieve_gpu(snapshot, *scorer, q, mismatched, 1, 32); });
    reject_input<std::invalid_argument>("predicate nonfinite", [&] { (void)retrieve_gpu(snapshot, *scorer, q, nonfinite, 1, 32); });
    reject_input<std::invalid_argument>("query before column", [&] { (void)retrieve_gpu(snapshot, *scorer, bad_query, bad_column, 1, 32); });
    reject_input<std::invalid_argument>("zero batch", [&] { (void)retrieve_gpu(snapshot, *scorer, q, {}, 1, 0); });
    const std::array blocked{Predicate{0, Comparison::less, std::int32_t{0}}, Predicate{2, Comparison::greater, 0.0f}};
#if defined(__x86_64__)
    const auto original = _mm_getcsr();
    _mm_setcsr(original & ~0x3fU);
#endif
    require(retrieve_gpu(snapshot, *scorer, q, blocked, 1, 32).empty(), "short circuit rejected rows");
#if defined(__x86_64__)
    const auto skipped = _mm_getcsr();
    _mm_setcsr(original);
    require((skipped & 2U) == 0, "skipped subnormal comparison touched MXCSR");
#endif
    scorer->close();
    scorer.reset();

    const float max = std::numeric_limits<float>::max();
    const std::array failure_values{1.0f, 0.0f, max / 2, max, max, 0.0f};
    const EmbeddingBuffer failure_embeddings(3, 2, std::vector<float>(failure_values.begin(), failure_values.end()));
    const ColumnBatch empty_columns({std::vector<std::int32_t>{1, 1, 1}});
    const std::array<RecordID, 3> ids{9, 3, 7};
    const ValidatedDataset failure_snapshot(empty_columns, failure_embeddings, ids);
    gpu::ResidentScorer failure(failure_values, 3, 2, 3, false);
    const std::array failure_query{2.0f, 1.0f};
    require(oracle_score(std::span(failure_values).subspan(2, 2), failure_query).first_failure == 1,
            "hand/independent failure dimension");
    bool caught = false;
    try { (void)retrieve_gpu(failure_snapshot, failure, failure_query, {}, 1, 2); }
    catch (const ScoreFailure& error) {
        require(error.slot == 1 && error.row == 1 && error.feature == 1, "reported earliest logical failure");
        std::cout << "expected numerical failure " << error.what() << '\n';
        caught = true;
    }
    require(caught, "whole query should fail");
    failure.close();
    std::cout << "PASS fixtures\n";
}

enum class Lifecycle { cpu, cold, warm, resident };

void lifecycle_check_cuda(cudaError_t status, const char* phase) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(phase) + ": " + cudaGetErrorName(status));
    }
}

CUcontext current_context() {
    // Resolve via the Runtime so this isolated binary needs no direct driver link.
    void* entry = nullptr;
    cudaDriverEntryPointQueryResult status{};
    lifecycle_check_cuda(cudaGetDriverEntryPoint("cuCtxGetCurrent", &entry, cudaEnableDefault, &status),
                         "resolve CUDA current-context query");
    require(status == cudaDriverEntryPointSuccess && entry != nullptr, "CUDA context query unavailable");
    CUcontext context = nullptr;
    const auto get_current = reinterpret_cast<decltype(&cuCtxGetCurrent)>(entry);
    require(get_current(&context) == CUDA_SUCCESS && context != nullptr, "no current CUDA context");
    return context;
}

void unrelated_cuda_completion() {
    // A checked, completed event proves the context can execute unrelated work
    // before dataset allocation. No scoring kernel or embedding transfer runs.
    cudaEvent_t event = nullptr;
    lifecycle_check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create prelude event");
    try {
        lifecycle_check_cuda(cudaEventRecord(event, nullptr), "submit prelude event");
        lifecycle_check_cuda(cudaEventSynchronize(event), "complete prelude event");
        lifecycle_check_cuda(cudaEventQuery(event), "verify prelude event completion");
    } catch (...) {
        // CUDA permits event destruction while pending. No buffer is attached.
        if (cudaEventDestroy(event) != cudaSuccess) {
            std::cerr << "prelude event cleanup unconfirmed\n";
            std::cerr.flush();
            std::_Exit(74);
        }
        throw;
    }
    lifecycle_check_cuda(cudaEventDestroy(event), "destroy prelude event");
}

struct RuntimeReady {
    CUcontext context = nullptr;
    Tick initialization = 0;
    Tick prelude = 0;
};

RuntimeReady prepare_runtime(bool prewarm) {
    const auto start = now_ns();
    gpu::select_device(false);
    const auto context = current_context();
    const auto initialized = now_ns();
    Tick prelude = 0;
    if (prewarm) {
        unrelated_cuda_completion();
        require(current_context() == context, "unrelated work changed the current CUDA context");
        prelude = now_ns() - initialized;
    }
    return {context, initialized - start, prelude};
}

struct SharedSetup {
    Tick construction, admission, release;
};

constexpr std::size_t lifecycle_max_queries = 128;

// All charged time is one continuous interval on the same host clock. The
// already-admitted CPU snapshot and fixed-size test sinks are shared prerequisites.
void lifecycle_session(const ValidatedDataset& snapshot, const Queries& queries, const Cell& cell,
                       const Golden& golden, const SharedSetup& shared, Lifecycle mode,
                       std::size_t q, std::size_t ordinal, RuntimeReady runtime = {}) {
    const bool device = mode != Lifecycle::cpu;
    const bool cold = mode == Lifecycle::cold;
    const bool warm = mode == Lifecycle::warm;
    std::array<Top, lifecycle_max_queries> consumed{};
    std::array<Tick, lifecycle_max_queries> starts{}, ends{};
    std::optional<gpu::ResidentScorer> scorer;
    gpu::ResidencyStats stats;
    if (device && !cold && !runtime.context) { runtime = prepare_runtime(true); }

    const auto setup_start = now_ns();
    if (cold) { runtime = prepare_runtime(false); }
    const auto packing_start = now_ns();
    auto packed = packing_start;
    if (device) {
        const auto flat = pack(snapshot);
        packed = now_ns();
        create_resident(scorer, snapshot, flat, stats);
    }
    const auto prepared = now_ns();

    const auto query_start = now_ns();
    for (std::size_t i = 0; i < q; ++i) {
        const auto index = i % queries.size();
        starts[i] = now_ns();
        {
            const auto result = scorer ? retrieve_gpu(snapshot, *scorer, queries[index], predicates, top_k, batch_size)
                                       : execute_batched(snapshot, queries[index], predicates, top_k, batch_size);
            require(result.size() == top_k, "lifecycle result size changed");
            std::copy(result.begin(), result.end(), consumed[i].begin());
        }
        ends[i] = now_ns();
    }
    const auto query_end = now_ns();
    if (scorer) {
        scorer->close();
        scorer.reset();
    }
    const auto cleaned = now_ns();

    // Compare every consumed Top-K after the charged interval. Gate A independently
    // verifies all candidate records for the same immutable snapshot and vectors.
    for (std::size_t i = 0; i < q; ++i) { exact(consumed[i], golden.expected.ranked[i % queries.size()]); }
    const auto after = device ? current_context() : nullptr;
    require(after == runtime.context, "CUDA context changed during the session");
    const bool charges_setup = cold || warm;
    const auto charged_begin = charges_setup ? setup_start : query_start;
    const auto charged_end = charges_setup ? cleaned : query_end;
    constexpr std::array names{"cpu", "cold", "warm", "resident"};
    std::cout << "{\"kind\":\"session\",\"mode\":\"" << names[static_cast<std::size_t>(mode)]
              << "\",\"cell\":\"" << cell.name << "\",\"ordinal\":" << ordinal << ",\"q\":" << q
              << ",\"data_fp\":" << golden.data_hash << ",\"query_fp\":" << golden.query_hash
              << ",\"exact\":true,\"runtime_was_ready\":" << (device && !cold ? "true" : "false")
              << ",\"context_before\":" << reinterpret_cast<std::uintptr_t>(runtime.context)
              << ",\"context_after\":" << reinterpret_cast<std::uintptr_t>(after)
              << ",\"context_init_ns\":" << runtime.initialization << ",\"prelude_completion_ns\":" << runtime.prelude
              << ",\"construction_ns\":" << shared.construction << ",\"admission_ns\":" << shared.admission
              << ",\"source_release_ns\":" << shared.release
              << ",\"pack_ns\":" << packed - packing_start << ",\"residency_ns\":" << (device ? prepared - packed : 0)
              << ",\"allocation_ns\":" << stats.allocation_ns << ",\"upload_ns\":" << stats.upload_ns
              << ",\"setup_start_ns\":" << setup_start << ",\"query_start_ns\":" << query_start
              << ",\"query_end_ns\":" << query_end << ",\"cleanup_end_ns\":" << cleaned
              << ",\"charged_begin_ns\":" << charged_begin << ",\"charged_end_ns\":" << charged_end
              << ",\"session_ns\":" << charged_end - charged_begin << ",\"teardown_ns\":" << (device ? cleaned - query_end : 0)
              << ",\"free_device_bytes\":" << stats.free_device_bytes << ",\"embedding_bytes\":" << stats.embedding_bytes
              << ",\"device_bytes\":" << stats.device_bytes << ",\"pinned_bytes\":" << stats.pinned_bytes << "}\n";
    for (std::size_t i = 0; i < q; ++i) {
        std::cout << "{\"kind\":\"query\",\"ordinal\":" << ordinal << ",\"index\":" << i
                  << ",\"vector\":" << i % queries.size() << ",\"start_ns\":" << starts[i]
                  << ",\"elapsed_ns\":" << ends[i] - starts[i] << "}\n";
    }
}

void lifecycle(const Cell& cell, const Golden& golden, Lifecycle mode, std::size_t q, bool repeat) {
    const auto start = now_ns();
    std::optional<Inputs> inputs(construct_inputs(cell));
    const auto queries = query_inputs();
    const auto generated = now_ns();
    require(input_fingerprint(*inputs) == golden.data_hash && query_fingerprint(queries) == golden.query_hash,
            "historical fingerprint changed");
    const auto admission = now_ns();
    const ValidatedDataset snapshot(inputs->columns, inputs->embeddings, inputs->ids);
    const auto admitted = now_ns();
    inputs.reset();
    const SharedSetup shared{generated - start, admitted - admission, now_ns() - admitted};
    if (repeat) {
        const auto runtime = prepare_runtime(true);
        lifecycle_session(snapshot, queries, cell, golden, shared, Lifecycle::warm, q, 0, runtime);
        lifecycle_session(snapshot, queries, cell, golden, shared, Lifecycle::resident, q, 1, runtime);
    } else {
        lifecycle_session(snapshot, queries, cell, golden, shared, mode, q, 0);
    }
}

} // namespace

#ifdef VECTORFORGE_PRIVATE_RETRIEVAL_REUSE
[[maybe_unused]] static int retrieval_main(int argc, char** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        const NearestEnvironment environment;
#if defined(__x86_64__)
        require((_mm_getcsr() & 0x8040U) == 0, "host FTZ/DAZ must be off");
#endif
        if (argc == 2 && std::string_view(argv[1]) == "--fixtures") { fixtures(); return 0; }
        require(argc >= 4, "usage: retrieval --fixtures | MODE CELL EXPECTED [ARGS]");
        const std::string_view mode = argv[1], cell_name = argv[2];
        const auto cell = std::ranges::find_if(cells, [&](const auto& item) { return item.name == cell_name; });
        require(cell != cells.end() && cell->rows == 100'000, "only the two historical N100K cells are supported");
        const auto golden = read_golden(argv[3], *cell);
        if (mode == "--gate" && argc == 5) { correctness(*cell, golden, argv[4]); }
        else if (mode == "--diagnose" && argc == 4) { measure(*cell, golden, true, true); }
        else if (mode == "--lifecycle-check" && argc == 4) { lifecycle(*cell, golden, Lifecycle::warm, 8, true); }
        else if (mode == "--lifecycle" && argc == 6) {
            const std::string_view path = argv[4], text = argv[5];
            constexpr std::array<std::string_view, 4> paths{"cpu", "cold", "warm", "resident"};
            const auto it = std::ranges::find(paths, path);
            require(it != paths.end(), "lifecycle path must be cpu, cold, warm or resident");
            std::size_t q = 0;
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), q);
            require(error == std::errc{} && end == text.data() + text.size() && q >= 1 && q <= lifecycle_max_queries,
                    "lifecycle Q must be an integer from 1 through 128");
            lifecycle(*cell, golden, static_cast<Lifecycle>(it - paths.begin()), q, false);
        }
        else if (mode == "--measure" && argc == 5) {
            const std::string_view path = argv[4];
            require(path == "cpu" || path == "gpu", "unknown measurement path");
            measure(*cell, golden, path == "gpu", false);
        } else { throw std::invalid_argument("invalid retrieval arguments"); }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "retrieval_failed: " << error.what() << '\n';
        return 1;
    }
}
