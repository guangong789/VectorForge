#define VECTORFORGE_PRIVATE_RETRIEVAL_REUSE
#include "retrieval.cpp"

#include <iomanip>
#include <thread>

namespace {

constexpr auto mixed_cell = cells[2];
constexpr std::size_t capacity = 34'000;
constexpr std::uint64_t data_fingerprint = 13302898193871766348ULL;
constexpr std::uint64_t vectors_fingerprint = 1173672798050634943ULL;
constexpr std::array<std::size_t, query_count> counts{1000, 33000, 33000, 114, 33000, capacity, 0, 1000};
constexpr std::array<std::uint64_t, query_count> candidate_fingerprints{
    12620881016167683185ULL, 14583374474110752340ULL, 847998124058146689ULL, 1051526958418800820ULL,
    4458426372241619427ULL, 13943246013302719162ULL, 12161962213042174405ULL, 12620881016167683185ULL};

const auto mixed_filters = [] {
    const Predicate tenant7{tenant_column, Comparison::equal, std::int32_t{7}};
    const Predicate tenant9{tenant_column, Comparison::equal, std::int32_t{9}};
    const Predicate cat2{category_column, Comparison::equal, std::int32_t{2}};
    const Predicate cat3{category_column, Comparison::equal, std::int32_t{3}};
    const Predicate recent{updated_column, Comparison::greater_equal, cutoff};
    return std::array<std::vector<Predicate>, query_count>{
        std::vector<Predicate>{tenant7, cat2, recent}, {tenant9, cat2, recent}, {tenant7, cat3, recent},
        {tenant7, cat2, {updated_column, Comparison::greater_equal, cutoff + 901}},
        {tenant7, cat2, {updated_column, Comparison::less, cutoff}},
        {tenant7, cat2}, {{tenant_column, Comparison::equal, std::int32_t{42}}}, {tenant7, cat2, recent}};
}();

std::uint32_t float_bits(float value) { return std::bit_cast<std::uint32_t>(value); }

std::uint64_t vector_fingerprint(const Query& query) {
    Fingerprint hash;
    for (const auto input : query) { hash.add(float_bits(input)); }
    return hash.value;
}

std::uint64_t predicate_fingerprint(std::span<const Predicate> filters) {
    Fingerprint hash;
    hash.add(filters.size());
    for (const auto& p : filters) {
        hash.add(p.column_index);
        hash.add(static_cast<unsigned>(p.comparison));
        hash.add(p.constant.index());
        std::visit([&]<typename T>(T value) {
            if constexpr (std::same_as<T, float>) { hash.add(float_bits(value)); }
            else { hash.add(static_cast<std::uint64_t>(value)); }
        }, p.constant);
    }
    return hash.value;
}

struct Identity {
    std::size_t count = 0;
    std::uint64_t predicates = 0, vector = 0, candidates = 0, scored = 0, top = 0;
};

void emit_identity(std::size_t q, const Identity& i) {
    std::cout << ",\"q\":" << q << ",\"m\":" << i.count << ",\"predicate_fp\":" << i.predicates
              << ",\"vector_fp\":" << i.vector << ",\"candidates_fp\":" << i.candidates
              << ",\"scored_fp\":" << i.scored << ",\"top_fp\":" << i.top;
}

struct Candidate {
    gpu::RowIndex row;
    RecordID id;
    gpu::Score score;
};
struct OracleQuery {
    Identity identity;
    std::vector<Candidate> candidates;
    std::vector<ScoredRecord> ordered;
};

bool oracle_matches(const Metadata& row, std::span<const Predicate> filters) {
    // The integration oracle reads typed AoS fields directly, with no ColumnBatch
    // binding or use of the GPU/CPU retrieval predicates.
    for (const auto& p : filters) {
        const auto actual = field(row, p.column_index);
        if (!std::visit([&]<class T>(T constant) {
                const auto* value = std::get_if<T>(&actual);
                require(value != nullptr, "oracle predicate type mismatch");
                return compare(*value, constant, p.comparison);
            }, p.constant)) { return false; }
    }
    return true;
}

std::array<OracleQuery, query_count> oracle(const LogicalData& source, const Queries& queries) {
    std::array<OracleQuery, query_count> result;
    for (std::size_t q = 0; q < result.size(); ++q) {
        auto& expected = result[q];
        const auto filters = std::span(mixed_filters[q]);
        for (std::size_t r = 0; r < source.rows.size(); ++r) {
            const auto& row = source.rows[r];
            if (!oracle_matches(row, filters)) { continue; }
            const auto score = oracle_score(std::span(source.embeddings).subspan(r * dimension, dimension), queries[q]);
            require(score.first_failure == gpu::no_failure, "oracle nonfinite intermediate value in mixed workload");
            expected.candidates.push_back({static_cast<gpu::RowIndex>(r), row.id, score});
            expected.ordered.push_back({row.id, score.value});
        }
        std::ranges::sort(expected.ordered, [](const ScoredRecord& a, const ScoredRecord& b) {
            return a.score > b.score || (a.score == b.score && a.id < b.id);
        });
        Fingerprint ids, scored;
        ids.add(expected.candidates.size());
        scored.add(expected.candidates.size());
        for (const auto& c : expected.candidates) {
            ids.add(c.id);
            scored.add(c.row);
            scored.add(c.id);
            scored.add(float_bits(c.score.value));
            scored.add(c.score.first_failure);
        }
        require(expected.candidates.size() == counts[q] && ids.value == candidate_fingerprints[q],
                "mixed oracle candidate identity differs from V2-04A");
        expected.identity = {counts[q], predicate_fingerprint(filters), vector_fingerprint(queries[q]), ids.value,
                             scored.value, result_fingerprint(std::span(expected.ordered).first(std::min(top_k, counts[q])))};
    }
    return result;
}

std::unique_ptr<const ValidatedDataset> admit(LogicalData&& source, Tick& elapsed) {
    const auto inputs = own_inputs(std::move(source));
    require(input_fingerprint(inputs) == data_fingerprint, "mixed dataset fingerprint changed");
    const auto start = now_ns();
    auto snapshot = std::make_unique<const ValidatedDataset>(inputs.columns, inputs.embeddings, inputs.ids);
    elapsed = now_ns() - start;
    return snapshot;
}

struct Residency {
    Tick context_ns = 0, pack_ns = 0, residency_ns = 0, event_setup_ns = 0, close_ns = 0;
    std::uintptr_t context = 0;
    int device = -1;
    std::size_t created = 0, destroyed = 0;
    gpu::ResidencyStats memory;
};

class MixedResident {
public:
    MixedResident(const ValidatedDataset& snapshot, std::size_t limit, Residency& timing, bool events = false)
        : snapshot_(snapshot), timing_(timing) {
        const auto runtime = prepare_runtime(false);
        context_ = runtime.context;
        timing_.context = reinterpret_cast<std::uintptr_t>(context_);
        lifecycle_check_cuda(cudaGetDevice(&timing_.device), "capture mixed owner device");
        device_ = timing_.device;
        timing_.context_ns = runtime.initialization;
        const auto start = now_ns();
        {
            const auto flat = pack(snapshot_);
            const auto packed = now_ns();
            timing_.pack_ns = packed - start;
            scorer_.emplace(flat, snapshot_.columns().row_count(), snapshot_.embeddings().dimension(), limit,
                            false, &timing_.memory);
            timing_.residency_ns = now_ns() - packed;
        }
        ++timing_.created;
        if (events) {
            const auto before = now_ns();
            scorer_->enable_device_timing();
            timing_.event_setup_ns = now_ns() - before;
        }
    }

    ~MixedResident() noexcept {
        if (!scorer_) { return; }
        try { close(); }
        catch (const std::exception& error) {
            std::fprintf(stderr, "mixed cleanup failed: %s\n", error.what());
            std::fflush(nullptr);
            std::_Exit(74);
        }
    }

    void close() {
        if (!scorer_) { return; }
        if (!context_matches()) {
            std::fputs("mixed cleanup context unconfirmed\n", stderr);
            std::fflush(nullptr);
            std::_Exit(74);
        }
        const auto start = now_ns();
        scorer_->close();
        scorer_.reset();
        ++timing_.destroyed;
        timing_.close_ns = now_ns() - start;
    }

    std::vector<ScoredRecord> query(std::span<const float> q, std::span<const Predicate> filters, std::size_t k,
                                    Diagnostic* phases = nullptr, CandidateTrace* trace = nullptr,
                                    MixedObservation* observation = nullptr) {
        check_owner();
        return retrieve_gpu(snapshot_, *scorer_, q, filters, k, batch_size, phases, trace, observation);
    }

    std::vector<gpu::Score> score_for_gate(std::span<const gpu::RowIndex> rows, std::span<const float> q) {
        check_owner();
        return scorer_->score(rows, q);
    }

private:
    bool context_matches() const noexcept {
        if (thread_ != std::this_thread::get_id()) { return false; }
        try {
            int device = -1;
            return cudaGetDevice(&device) == cudaSuccess && device == device_ && current_context() == context_;
        } catch (const std::exception&) {
            return false;
        }
    }

    void check_owner() const {
        if (!scorer_ || !context_matches()) { throw std::logic_error("mixed owner closed or context changed"); }
    }

    const ValidatedDataset& snapshot_;
    Residency& timing_;
    const std::thread::id thread_ = std::this_thread::get_id();
    CUcontext context_ = nullptr;
    int device_ = -1;
    std::optional<gpu::ResidentScorer> scorer_;
};

void check_trace(const CandidateTrace& trace, const OracleQuery& expected, const ValidatedDataset& data) {
    require(trace.rows.size() == expected.candidates.size() && trace.scores.size() == trace.rows.size(),
            "wrong mixed candidate or score count");
    for (std::size_t i = 0; i < trace.rows.size(); ++i) {
        const auto& c = expected.candidates[i];
        require(trace.rows[i] == c.row && data.record_ids()[trace.rows[i]] == c.id, "mixed row/ID order mismatch");
        require(float_bits(trace.scores[i].value) == float_bits(c.score.value) &&
                trace.scores[i].first_failure == c.score.first_failure, "mixed candidate score/status mismatch");
    }
}

void emit_candidates(std::size_t q, const CandidateTrace& trace, const ValidatedDataset& data) {
    std::cout << "{\"kind\":\"candidate_trace\",\"q\":" << q << ",\"slots\":[";
    for (std::size_t i = 0; i < trace.rows.size(); ++i) {
        std::cout << (i ? ",[" : "[") << trace.rows[i] << ',' << data.record_ids()[trace.rows[i]] << ','
                  << float_bits(trace.scores[i].value) << ',' << trace.scores[i].first_failure << ']';
    }
    std::cout << "]}\n";
}

void gate(bool device, bool emit_slots) {
    auto source = generate(mixed_cell);
    const auto queries = query_inputs();
    require(query_fingerprint(queries) == vectors_fingerprint, "query sequence changed");
    const auto start = now_ns();
    const auto expected = oracle(source, queries);
    const auto oracle_ns = now_ns() - start;
    std::array<std::vector<ScoredRecord>, query_count> retained;
    Residency residency;
    std::array<std::vector<gpu::RowIndex>, query_count> routes;
    {
        Tick admission_ns = 0;
        const auto snapshot = admit(std::move(source), admission_ns);
        std::optional<MixedResident> owner;
        if (device) { owner.emplace(*snapshot, capacity, residency); }
        for (std::size_t q = 0; q < queries.size(); ++q) {
            const auto& filters = mixed_filters[q];
            if (device) {
                CandidateTrace trace;
                retained[q] = owner->query(queries[q], filters, top_k, nullptr, &trace);
                check_trace(trace, expected[q], *snapshot);
                if (emit_slots) { emit_candidates(q, trace, *snapshot); }
                routes[q] = std::move(trace.rows);
            } else {
                // K=N returns all ordered candidates without adding a hook to Contract S.
                const auto cpu = execute_batched(*snapshot, queries[q], filters, mixed_cell.rows, batch_size);
                const auto scalar = execute_scalar_reference(*snapshot, queries[q], filters, mixed_cell.rows);
                exact(cpu, expected[q].ordered);
                exact(scalar, expected[q].ordered);
                retained[q] = execute_batched(*snapshot, queries[q], filters, top_k, batch_size);
            }
        }
        if (owner) { owner->close(); }
    }
    if (device) {
        std::vector<gpu::RowIndex> intersection;
        std::ranges::set_intersection(routes[1], routes[2], std::back_inserter(intersection));
        require(intersection.empty() && routes[0] == routes[7] && queries[0] != queries[7] &&
                routes[5].size() == capacity && routes[6].empty() && !routes[7].empty(),
                "mixed GPU transition identity changed");
        require(residency.created == 1 && residency.destroyed == 1, "mixed owner lifetime differs");
        std::cout << "{\"kind\":\"transitions\",\"q1_q2_intersection\":" << intersection.size()
                  << ",\"q0_q7_rows_equal\":true,\"q0_q7_vectors_distinct\":true,"
                     "\"q5_q6_q7\":[34000,0,1000],\"owners_created\":1,\"owners_destroyed\":1}\n";
    }
    for (std::size_t q = 0; q < queries.size(); ++q) {
        const auto k = std::min(top_k, counts[q]);
        exact(retained[q], std::span(expected[q].ordered).first(k));
        std::cout << "{\"kind\":\"verified_query\"";
        emit_identity(q, expected[q].identity);
        std::cout << ",\"top_size\":" << retained[q].size() << "}\n";
    }
    std::cout << "{\"kind\":\"gate\",\"device\":" << (device ? "true" : "false")
              << ",\"dataset_fp\":" << data_fingerprint << ",\"query_sequence_fp\":" << vectors_fingerprint
              << ",\"oracle_ns\":" << oracle_ns << ",\"owning_results_verified_after_close\":true,\"pass\":true}\n";
}

void boundaries() {
    const float high = std::numeric_limits<float>::max();
    const ColumnBatch columns({std::vector<std::int32_t>{1, 2, 3, 4}});
    const EmbeddingBuffer embeddings(4, 2, {1.0f, 0.0f, 1.0f, -0.0f, high, 0.0f, high, 0.0f});
    const std::array<RecordID, 4> ids{11, 7, 21, 19};
    const ValidatedDataset data(columns, embeddings, ids);
    Residency timing;
    MixedResident owner(data, 2, timing);
    const std::array q{2.0f, 1.0f};
    const std::array zero{Predicate{0, Comparison::less, std::int32_t{1}}};
    const std::array first_two{Predicate{0, Comparison::less, std::int32_t{3}}};
    const std::array last_two{Predicate{0, Comparison::greater, std::int32_t{2}}};
    const std::array one{Predicate{0, Comparison::equal, std::int32_t{2}}};
    const std::array type_error{Predicate{0, Comparison::equal, 1.0f}};
    const std::array<gpu::RowIndex, 1> bad_row{4};
    const std::array bad_query{std::numeric_limits<float>::quiet_NaN(), 1.0f};

    const auto without_submission = [](const char* name, auto&& verify) {
        std::cerr << "mixed-trace " << name << " begin\n";
        verify();
        std::cerr << "mixed-trace " << name << " end\n";
    };
    without_submission("zero", [&] { require(owner.query(q, zero, 1).empty(), "M=0 result"); });
    without_submission("over-capacity", [&] {
        reject_input<std::invalid_argument>("M>C", [&] { (void)owner.query(q, {}, 1); });
    });
    without_submission("bad-row", [&] {
        reject_input<std::out_of_range>("RowIndex", [&] { (void)owner.score_for_gate(bad_row, q); });
    });
    without_submission("query-shape", [&] {
        reject_input<std::invalid_argument>("dimension", [&] { (void)owner.query(std::span(q).first(1), one, 1); });
        reject_input<std::invalid_argument>("nonfinite query", [&] { (void)owner.query(bad_query, one, 1); });
    });
    without_submission("zero-k", [&] {
        reject_input<std::invalid_argument>("invalid predicate at K=0", [&] { (void)owner.query(q, type_error, 0); });
        require(owner.query(q, {}, 0).empty(), "validated K=0 result");
    });

    exact(owner.query(q, one, 3), std::array<ScoredRecord, 1>{{{7, 2.0f}}});
    exact(owner.query(q, first_two, 3), std::array<ScoredRecord, 2>{{{7, 2.0f}, {11, 2.0f}}});
    try {
        (void)owner.query(q, last_two, 1);
        throw std::logic_error("overflow published a Top-K");
    } catch (const ScoreFailure& failure) {
        require(failure.slot == 0 && failure.row == 2 && failure.feature == 0, "wrong first G failure slot");
    }
    owner.close();
    owner.close();
    reject_input<std::logic_error>("mixed owner closed", [&] { (void)owner.query(q, one, 1); });
    require(timing.created == 1 && timing.destroyed == 1, "mixed double close changed owner counts");
    std::cout << "{\"kind\":\"boundary_gate\",\"m0\":true,\"m1\":true,\"m_eq_c\":true,"
                 "\"invalid_continuation\":true,\"overflow\":true,\"pass\":true}\n";
}

std::array<Identity, query_count> load_expected(const char* path, const Queries& queries) {
    std::ifstream file(path);
    std::array<Identity, query_count> rows{};
    for (std::size_t q = 0; q < rows.size(); ++q) {
        auto& i = rows[q];
        require(static_cast<bool>(file >> i.count >> i.predicates >> i.vector >> i.candidates >> i.scored >> i.top),
                "invalid frozen mixed result manifest");
        require(i.count == counts[q] && i.candidates == candidate_fingerprints[q] &&
                i.vector == vector_fingerprint(queries[q]) && i.predicates == predicate_fingerprint(mixed_filters[q]),
                "frozen mixed manifest identity differs");
    }
    std::string extra;
    require(!(file >> extra), "trailing mixed manifest data");
    return rows;
}

struct QuerySample {
    Diagnostic phases;
    gpu::DevicePhases events;
    MixedObservation observations;
    Tick wall_ns = 0, consume_ns = 0, destroy_ns = 0;
    Top consumed{};
    std::size_t size = 0;
};

void emit_stages(const QuerySample& s) {
    const auto& p = s.phases;
    const auto& g = p.scoring;
    std::cout << ",\"validate_ns\":" << p.validate << ",\"reservation_ns\":" << p.row_reserve
              << ",\"scan_emit_ns\":" << p.scan_emit << ",\"prepare_ns\":" << g.prepare_ns
              << ",\"h2d_submit_ns\":" << g.h2d_submit_ns << ",\"h2d_wait_ns\":" << g.h2d_wait_ns
              << ",\"kernel_submit_ns\":" << g.kernel_submit_ns << ",\"kernel_wait_ns\":" << g.kernel_wait_ns
              << ",\"d2h_submit_ns\":" << g.d2h_submit_ns << ",\"d2h_wait_ns\":" << g.d2h_wait_ns
              << ",\"unpack_ns\":" << g.unpack_ns << ",\"rank_ns\":" << p.rank
              << ",\"verify_ns\":" << s.observations.verify_ns << ",\"map_ns\":" << s.observations.map_ns
              << ",\"topk_ns\":" << s.observations.topk_ns << ",\"h2d_event_ms\":" << s.events.h2d_ms
              << ",\"kernel_event_ms\":" << s.events.kernel_ms << ",\"d2h_event_ms\":" << s.events.d2h_ms
              << ",\"events_read_ns\":" << s.events.read_ns;
}

void measure_mixed(const char* golden, bool device, bool diagnose, bool trace) {
    const auto start = now_ns();
    auto source = generate(mixed_cell);
    const auto queries = query_inputs();
    const auto generation_ns = now_ns() - start;
    require(query_fingerprint(queries) == vectors_fingerprint, "query sequence changed");
    const auto expected = load_expected(golden, queries);
    const auto admission_begin = now_ns();
    Tick admission_ns = 0;
    const auto snapshot = admit(std::move(source), admission_ns);
    const auto setup_ns = now_ns() - admission_begin - admission_ns;
    // Metadata is only needed by the gate oracle.
    std::vector<Metadata>{}.swap(source.rows);
    Residency timing;
    std::optional<MixedResident> owner;
    if (device) { owner.emplace(*snapshot, capacity, timing, diagnose); }
    std::array<QuerySample, query_count> samples{};
    const auto session_begin = now_ns();
    for (std::size_t q = 0; q < queries.size(); ++q) {
        auto& sample = samples[q];
        if (diagnose) { sample.observations.events = &sample.events; }
        if (trace) { std::cerr << "mixed-trace q" << q << " begin\n"; }
        const auto query_begin = now_ns();
        Tick consumed = 0;
        {
            const auto result = device ? owner->query(queries[q], mixed_filters[q], top_k,
                                diagnose ? &sample.phases : nullptr, nullptr, &sample.observations)
                                       : execute_batched(*snapshot, queries[q], mixed_filters[q], top_k, batch_size);
            const auto consume = now_ns();
            require(result.size() <= sample.consumed.size(), "unexpected result size");
            sample.size = result.size();
            std::ranges::copy(result, sample.consumed.begin());
            consumed = now_ns();
            sample.consume_ns = consumed - consume;
        }
        const auto end = now_ns();
        sample.wall_ns = end - query_begin;
        sample.destroy_ns = end - consumed;
        if (trace) { std::cerr << "mixed-trace q" << q << " end\n"; }
    }
    const auto session_ns = now_ns() - session_begin;
    if (owner) { owner->close(); }

    for (std::size_t q = 0; q < samples.size(); ++q) {
        const auto& sample = samples[q];
        require(sample.size == std::min(top_k, expected[q].count) &&
                result_fingerprint(std::span(sample.consumed).first(sample.size)) == expected[q].top,
                "timed mixed Top-K fingerprint changed");
        require(!device || sample.observations.matched == expected[q].count, "timed actual candidate count changed");
    }
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
              << "{\"kind\":\"session\",\"route\":\"" << (device ? "gpu" : "cpu")
              << "\",\"diagnostic\":" << (diagnose ? "true" : "false") << ",\"session_ns\":" << session_ns
              << ",\"dataset_fp\":" << data_fingerprint << ",\"query_sequence_fp\":" << vectors_fingerprint
              << ",\"generation_ns\":" << generation_ns << ",\"fingerprint_and_inputs_ns\":" << setup_ns
              << ",\"admission_ns\":" << admission_ns << ",\"context_ns\":" << timing.context_ns
              << ",\"pack_ns\":" << timing.pack_ns << ",\"resident_ns\":" << timing.residency_ns
              << ",\"allocation_ns\":" << timing.memory.allocation_ns << ",\"upload_ns\":" << timing.memory.upload_ns
              << ",\"event_setup_ns\":" << timing.event_setup_ns << ",\"close_ns\":" << timing.close_ns
              << ",\"device_bytes\":" << timing.memory.device_bytes << ",\"pinned_bytes\":" << timing.memory.pinned_bytes
              << ",\"free_bytes\":" << timing.memory.free_device_bytes << ",\"context\":" << timing.context
              << ",\"device\":" << timing.device << ",\"owners_created\":" << timing.created
              << ",\"owners_destroyed\":" << timing.destroyed << ",\"pass\":true}\n";
    for (std::size_t q = 0; q < samples.size(); ++q) {
        const auto& s = samples[q];
        std::cout << "{\"kind\":\"query\"";
        emit_identity(q, expected[q]);
        std::cout << ",\"top_size\":" << s.size << ",\"wall_ns\":" << s.wall_ns
                  << ",\"consume_ns\":" << s.consume_ns << ",\"destroy_ns\":" << s.destroy_ns;
        if (device) { std::cout << ",\"actual_m\":" << s.observations.matched; }
        if (diagnose) { emit_stages(s); }
        std::cout << "}\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const NearestEnvironment rounding;
#if defined(__x86_64__)
        require((_mm_getcsr() & 0x8040U) == 0, "mixed workload requires FTZ/DAZ off");
#endif
        require(argc >= 2, "usage: mixed retrieval requires a mode");
        const std::string_view mode = argv[1];
        if (argc == 2 && mode == "--gate-cpu") { gate(false, false); }
        else if (mode == "--gate-g" && (argc == 2 || (argc == 3 && std::string_view(argv[2]) == "--emit-slots"))) {
            gate(true, argc == 3);
        } else if (argc == 2 && mode == "--boundaries-g") { boundaries(); }
        else {
            require(argc == 3 && (mode == "--cpu-s" || mode == "--gpu-g" || mode == "--diagnostic-g" || mode == "--trace-g"),
                    "explicit S/G mode and frozen result manifest required");
            measure_mixed(argv[2], mode != "--cpu-s", mode == "--diagnostic-g", mode == "--trace-g");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mixed_retrieval_failed: " << error.what() << '\n';
        return 1;
    }
}
