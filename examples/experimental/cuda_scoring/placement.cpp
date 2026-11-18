#include "placement_cuda.hpp"

#include <dlfcn.h>
#include <sys/utsname.h>

#define main historical_workload_entry
#include "../../integration/ai_retrieval/workload.cpp"
#undef main

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace {
namespace cuda = placement_cuda;

constexpr std::uint64_t s90_data = 9034647294887284646ULL;
constexpr std::uint64_t historical_queries = 1173672798050634943ULL;
constexpr std::string_view evidence_id = "H0-03/20261118T072526.128460003Z/N100K-S90/resident/Q8";

class Workload {
public:
    static Workload historical(bool dense) {
        // Only this factory confers evidence provenance. It accepts no user-
        // supplied payloads or digests. The deep copy seals the generated input.
        const auto inputs = construct_inputs(cells[dense ? 3 : 2]);
        const auto queries = query_inputs();
        const auto hash = input_fingerprint(inputs);
        require(hash == (dense ? s90_data : 13302898193871766348ULL) &&
                query_fingerprint(queries) == historical_queries, "historical generator identity changed");
        return Workload(inputs, queries, dense);
    }

    const std::shared_ptr<const ValidatedDataset> snapshot;
    const Queries queries;
    const bool dense;

private:
    Workload(const Inputs& inputs, Queries ordered, bool is_dense)
        : snapshot(std::make_shared<const ValidatedDataset>(inputs.columns, inputs.embeddings, inputs.ids)),
          queries(std::move(ordered)), dense(is_dense) {}
};

bool host_matches_evidence() {
    utsname host{};
    if (uname(&host) != 0 || std::string_view(host.machine) != "x86_64" ||
        std::string_view(host.sysname) != "Linux" ||
        std::string_view(host.release) != "6.18.33.2-microsoft-standard-WSL2") { return false; }
    std::ifstream cpu("/proc/cpuinfo");
    for (std::string line; std::getline(cpu, line);) {
        if (line.starts_with("model name")) { return line.ends_with("12th Gen Intel(R) Core(TM) i9-12900H"); }
    }
    return false;
}

bool common_fp_domain() {
#if defined(__x86_64__)
    return std::fegetround() == FE_TONEAREST && (_mm_getcsr() & 0x8040U) == 0;
#else
    return false;
#endif
}

class Module {
public:
    Module() {
        require(host_matches_evidence(), "host environment is outside placement evidence");
        const auto library = std::filesystem::read_symlink("/proc/self/exe").parent_path() /
                             "libvectorforge_placement_cuda.so";
        // Keep exception types and their unwind information mapped if a GPU
        // exception outlives the optional resident object.
        handle_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
        if (!handle_) { throw std::runtime_error(std::string("load placement CUDA module: ") + dlerror()); }
    }
    ~Module() { (void)dlclose(handle_); }
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    const cuda::Api& api() const {
        const auto symbol = dlsym(handle_, "vectorforge_placement_cuda_api");
        require(symbol != nullptr, "private CUDA module entry point missing");
        return *reinterpret_cast<decltype(&vectorforge_placement_cuda_api)>(symbol)();
    }

private:
    void* handle_;
};

class ResidentHandle {
public:
    explicit ResidentHandle(const Workload& work, std::size_t capacity)
        : api_(module_.api()), resident_(api_.create(work.snapshot, capacity, prep_)), evidenced_(work.dense) {}
    ~ResidentHandle() { api_.destroy(resident_); }
    ResidentHandle(const ResidentHandle&) = delete;
    ResidentHandle& operator=(const ResidentHandle&) = delete;
    ResidentHandle(ResidentHandle&&) = delete;
    ResidentHandle& operator=(ResidentHandle&&) = delete;

    bool owns(const std::shared_ptr<const ValidatedDataset>& snapshot) const noexcept {
        return api_.owns(*resident_, snapshot);
    }
    bool evidenced() const noexcept { return evidenced_; }
    cuda::State status() const noexcept { return api_.status(*resident_); }
    const cuda::Preparation& preparation() const noexcept { return prep_; }
    void close() { api_.close(*resident_); }
    std::vector<ScoredRecord> query(std::span<const float> q, std::span<const Predicate> filters,
                                    std::size_t k, std::size_t batch, cuda::CandidateEvidence* bits) {
        return api_.query(*resident_, q, filters, k, batch, bits);
    }

private:
    Module module_;
    const cuda::Api& api_;
    cuda::Preparation prep_;
    cuda::Resident* resident_;
    const bool evidenced_;
};

enum class Consent { unspecified, strict_s, accept_g };
enum class Backend { none, cpu, gpu };
enum class Reason { strict_cpu, consent_missing, residency_unavailable, identity_mismatch, closed_or_invalidated,
                    device_context_unsupported, capacity_insufficient, evidence_mismatch, unsupported_q, verified_evidence };
constexpr std::array backend_names{"none", "cpu", "gpu"};
constexpr std::array reason_names{"strict_cpu", "consent_missing", "residency_unavailable", "identity_mismatch",
    "closed_or_invalidated", "device_context_unsupported", "capacity_insufficient", "evidence_mismatch",
    "unsupported_q", "verified_evidence"};

struct Request {
    std::shared_ptr<const ValidatedDataset> snapshot;
    std::span<const Query> queries;
    std::span<const Predicate> filters;
    std::size_t k, batch;
    std::optional<std::size_t> declared;
    Consent consent;
    bool allow_cpu;
};

struct Decision {
    Backend backend;
    Reason reason;
    bool cpu_fallback_not_authorized = false;
    bool operator==(const Decision&) const = default;
};

bool exact_query_evidence(std::span<const Query> actual, const Queries& expected) {
    return std::ranges::equal(actual, expected, [](const Query& lhs, const Query& rhs) {
        return std::ranges::equal(lhs, rhs, [](float a, float b) {
            return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
        });
    });
}

bool exact_predicate_evidence(std::span<const Predicate> actual) {
    return std::ranges::equal(actual, predicates, [](const auto& a, const auto& b) {
        return a.column_index == b.column_index && a.comparison == b.comparison && a.constant == b.constant;
    });
}

Decision choose(const Workload& fixture, const Request& request, const ResidentHandle* resident) {
    if (request.consent == Consent::strict_s) { return {Backend::cpu, Reason::strict_cpu}; }
    if (request.consent != Consent::accept_g) { return {Backend::none, Reason::consent_missing}; }
    const auto decline = [&](Reason why) {
        return Decision{request.allow_cpu ? Backend::cpu : Backend::none, why, !request.allow_cpu};
    };
    if (request.declared != query_count || request.queries.size() != query_count) {
        return decline(Reason::unsupported_q);
    }
    if (!fixture.dense || request.snapshot->columns().row_count() != 100'000 ||
        request.snapshot->embeddings().dimension() != dimension || request.k != top_k || request.batch != batch_size ||
        !exact_predicate_evidence(request.filters) || !exact_query_evidence(request.queries, fixture.queries)) {
        return decline(Reason::evidence_mismatch);
    }
    if (!resident) { return decline(Reason::residency_unavailable); }
    if (!resident->evidenced() || !resident->owns(request.snapshot)) { return decline(Reason::identity_mismatch); }
    if (!common_fp_domain()) { return decline(Reason::device_context_unsupported); }
    switch (resident->status()) {
        case cuda::State::ready: return {Backend::gpu, Reason::verified_evidence};
        case cuda::State::closed:
        case cuda::State::invalid: return decline(Reason::closed_or_invalidated);
        case cuda::State::device_context: return decline(Reason::device_context_unsupported);
        case cuda::State::capacity: return decline(Reason::capacity_insufficient);
    }
    throw std::logic_error("unrecognized placement state");
}

std::vector<ScoredRecord> execute_query(Backend backend, const Request& request, std::span<const float> q,
                                       ResidentHandle* resident, cuda::CandidateEvidence* evidence = nullptr) {
    if (backend == Backend::cpu) {
        return execute_batched(*request.snapshot, q, request.filters, request.k, request.batch);
    }
    require(backend == Backend::gpu && resident, "no selected execution path");
    return resident->query(q, request.filters, request.k, request.batch, evidence);
}

Request request_for(const Workload& work, Consent consent, bool allow_cpu = false) {
    return {work.snapshot, work.queries, predicates, top_k, batch_size, query_count, consent, allow_cpu};
}

void emit_decision(std::ostream& out, Decision d) {
    out << "\"backend\":\"" << backend_names[static_cast<std::size_t>(d.backend)]
        << "\",\"reason\":\"" << reason_names[static_cast<std::size_t>(d.reason)]
        << "\",\"cpu_fallback_not_authorized\":" << (d.cpu_fallback_not_authorized ? "true" : "false");
}

void check_decision(std::string_view name, Decision actual, Decision expected) {
    require(actual == expected, "placement decision mismatch");
    std::cout << "{\"kind\":\"decision\",\"case\":\"" << name << "\",";
    emit_decision(std::cout, actual);
    std::cout << "}\n";
}

struct PlacementSession {
    Decision decision{};
    Tick total = 0, selection = 0;
    std::array<Tick, query_count> times{};
    std::array<Top, query_count> consumed{};
    std::vector<std::vector<ScoredRecord>> other_results;
};

PlacementSession run_session(const Workload& work, const Request& request, ResidentHandle* resident,
                             std::array<cuda::CandidateEvidence, query_count>* evidence = nullptr,
                             bool historical_sink = true) {
    // This experimental session runner has eight fixed result sinks. The CPU
    // query adapter also serves the strict-path validation tests directly.
    PlacementSession session;
    const auto begin = now_ns();
    session.decision = choose(work, request, resident);
    session.selection = now_ns() - begin;
    if (session.decision.backend == Backend::none) { return session; }
    historical_sink = historical_sink && request.queries.size() == query_count && request.k == top_k;
    for (std::size_t i = 0; i < request.queries.size(); ++i) {
        const auto start = now_ns();
        {
            auto* trace = evidence && i < evidence->size() ? &(*evidence)[i] : nullptr;
            auto owned = execute_query(session.decision.backend, request, request.queries[i], resident, trace);
            if (historical_sink && owned.size() == top_k) {
                std::copy(owned.begin(), owned.end(), session.consumed[i].begin());
            } else {
                session.other_results.push_back(std::move(owned));
            }
        }
        if (i < session.times.size()) { session.times[i] = now_ns() - start; }
    }
    session.total = now_ns() - begin;
    return session;
}

void verify_top(const PlacementSession& session, const Golden& golden) {
    require(session.decision.backend != Backend::none && session.total > 0, "session did not execute");
    for (std::size_t i = 0; i < session.consumed.size(); ++i) { exact(session.consumed[i], golden.expected.ranked[i]); }
}

void verify_candidates(const Workload& work, const Golden& golden, const char* path,
                       const std::array<cuda::CandidateEvidence, query_count>& evidence) {
    const auto oracle = generate(cells[3]);
    std::ofstream raw(path, std::ios::binary | std::ios::trunc);
    require(raw.good(), "cannot create placement score evidence");
    for (std::size_t j = 0; j < query_count; ++j) {
        const auto& trace = evidence[j];
        const auto& q = work.queries[j];
        require(trace.rows.size() == cells[3].matches && trace.scores.size() == trace.rows.size(), "candidate count");
        std::size_t slot = 0;
        for (std::size_t row = 0; row < oracle.rows.size(); ++row) {
            if (!matches(oracle.rows[row])) { continue; }
            require(slot < trace.rows.size() && trace.rows[slot] == row &&
                    work.snapshot->record_ids()[row] == oracle.rows[row].id, "candidate identity");
            float score = 0.0f;
            for (std::size_t d = 0; d < q.size(); ++d) {
                score = std::fma(q[d], oracle.embeddings[row * dimension + d], score);
                require(std::isfinite(score), "unexpected placement oracle prefix");
            }
            const auto actual = trace.scores[slot++];
            const auto id = oracle.rows[row].id;
            const auto cpu = std::bit_cast<std::uint32_t>(score), gpu = std::bit_cast<std::uint32_t>(actual.value);
            require(cpu == gpu && actual.first_failure == cuda_scoring_pilot::no_failure, "placement score/status mismatch");
            const std::array words{static_cast<std::uint32_t>(row), static_cast<std::uint32_t>(id),
                                   static_cast<std::uint32_t>(id >> 32), cpu, gpu,
                                   cuda_scoring_pilot::no_failure, actual.first_failure};
            static_assert(std::endian::native == std::endian::little);
            raw.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(sizeof(words)));
        }
        require(slot == trace.rows.size(), "candidate enumeration count");
        exact(execute_batched(*work.snapshot, q, predicates, top_k, batch_size), golden.expected.ranked[j]);
    }
    raw.close();
    require(raw.good(), "cannot finish placement score evidence");
}

void emit_measurement(const PlacementSession& samples, const cuda::Preparation& prep) {
    std::cout << "{\"kind\":\"measurement\",\"evidence\":\"" << evidence_id << "\",";
    emit_decision(std::cout, samples.decision);
    std::cout << ",\"session_ns\":" << samples.total << ",\"decision_ns\":" << samples.selection
              << ",\"context_ns\":" << prep.context_ns << ",\"pack_ns\":" << prep.pack_ns
              << ",\"residency_ns\":" << prep.residency_ns << ",\"allocation_ns\":" << prep.memory.allocation_ns
              << ",\"upload_ns\":" << prep.memory.upload_ns << ",\"device\":" << prep.device
              << ",\"context\":" << prep.context << ",\"device_bytes\":" << prep.memory.device_bytes
              << ",\"pinned_bytes\":" << prep.memory.pinned_bytes << ",\"free_bytes\":" << prep.memory.free_device_bytes
              << ",\"embedding_bytes\":" << prep.memory.embedding_bytes << ",\"exact\":true,\"query_ns\":[";
    for (std::size_t i = 0; i < samples.times.size(); ++i) { std::cout << (i ? "," : "") << samples.times[i]; }
    std::cout << "]}\n";
}

void cpu_tests() {
    const auto work = Workload::historical(true);
    auto request = request_for(work, Consent::strict_s);
    const auto strict = Decision{Backend::cpu, Reason::strict_cpu};
    check_decision("strict_no_handle", choose(work, request, nullptr), strict);
    const auto saved_rounding = std::fegetround();
    require(std::fesetround(FE_DOWNWARD) == 0, "cannot set CPU test rounding");
    request.batch = 0;
    try {
        (void)execute_query(choose(work, request, nullptr).backend, request, {}, nullptr);
        throw std::runtime_error("strict invalid batch was accepted");
    } catch (const std::invalid_argument& error) {
        require(std::string_view(error.what()) == "batch size must be positive", "strict validation precedence");
    }
    require(std::fegetround() == FE_DOWNWARD, "strict path changed rounding mode");
    require(std::fesetround(saved_rounding) == 0, "cannot restore CPU test rounding");
    request = request_for(work, Consent::strict_s);
    const auto good = execute_query(choose(work, request, nullptr).backend, request, work.queries[0], nullptr);
    exact(good, execute_batched(*work.snapshot, work.queries[0], predicates, top_k, batch_size));
    check_decision("strict_executed", strict, strict);
    auto absent = request_for(work, Consent::unspecified, true);
    check_decision("no_consent", choose(work, absent, nullptr), {Backend::none, Reason::consent_missing});
    auto gpu = request_for(work, Consent::accept_g);
    check_decision("no_residency_g_only", choose(work, gpu, nullptr), {Backend::none, Reason::residency_unavailable, true});
    gpu.allow_cpu = true;
    check_decision("cpu_alternative_authorized", choose(work, gpu, nullptr), {Backend::cpu, Reason::residency_unavailable});
    gpu.queries = gpu.queries.first(1);
    gpu.declared = 1;
    const auto chosen = run_session(work, gpu, nullptr, nullptr, false);
    check_decision("cpu_alternative_executed", chosen.decision, {Backend::cpu, Reason::unsupported_q});
    require(chosen.other_results.size() == 1, "CPU alternative did not run");
    exact(chosen.other_results[0], good);
    std::cout << "PASS cpu placement\n";
}

void gpu_decision_tests(const Workload& work, ResidentHandle& resident) {
    const auto accepted = request_for(work, Consent::accept_g);
    const auto rejected = [&](std::string_view name, const Request& r, Reason why) {
        check_decision(name, choose(work, r, &resident), {Backend::none, why, true});
    };
    check_decision("historical_gpu", choose(work, accepted, &resident), {Backend::gpu, Reason::verified_evidence});
    for (const std::size_t q : {1U, 7U, 9U}) {
        auto changed = accepted;
        changed.declared = q;
        rejected("q" + std::to_string(q), changed, Reason::unsupported_q);
    }
    {
        auto changed = accepted;
        changed.declared.reset();
        rejected("unknown_q", changed, Reason::unsupported_q);
        changed = accepted;
        changed.queries = changed.queries.first(7);
        rejected("actual_q7", changed, Reason::unsupported_q);
        changed = accepted;
        changed.k = 19;
        rejected("changed_k", changed, Reason::evidence_mismatch);
        changed = accepted;
        changed.batch = 17;
        rejected("changed_batch", changed, Reason::evidence_mismatch);
    }
    {
        auto altered = predicates;
        auto changed = accepted;
        changed.filters = altered;
        altered[0].constant = std::int32_t{8};
        rejected("predicate_value", changed, Reason::evidence_mismatch);
        altered = predicates;
        altered[0].comparison = Comparison::not_equal;
        rejected("predicate_operator", changed, Reason::evidence_mismatch);
        altered = predicates;
        std::swap(altered[0], altered[1]);
        rejected("predicate_order", changed, Reason::evidence_mismatch);
    }
    {
        auto altered = work.queries;
        auto changed = accepted;
        changed.queries = altered;
        altered[0][0] = std::nextafter(altered[0][0], 1.0f);
        rejected("query_bits", changed, Reason::evidence_mismatch);
        altered = work.queries;
        std::swap(altered[0], altered[1]);
        rejected("query_order", changed, Reason::evidence_mismatch);
    }
    {
        const auto sparse = Workload::historical(false);
        check_decision("s1", choose(sparse, request_for(sparse, Consent::accept_g), &resident),
                       {Backend::none, Reason::evidence_mismatch, true});
        rejected("s1_substituted", request_for(sparse, Consent::accept_g), Reason::identity_mismatch);
    }
    {
        const auto twin = Workload::historical(true);
        rejected("identical_content_other_owner", request_for(twin, Consent::accept_g), Reason::identity_mismatch);
        const std::shared_ptr<const ValidatedDataset> forged(work.snapshot.get(), [](const ValidatedDataset*) {});
        const Request alias{forged, work.queries, predicates, top_k, batch_size, query_count, Consent::accept_g, false};
        rejected("same_address_other_owner", alias, Reason::identity_mismatch);
    }
    {
        auto input = construct_inputs(cells[3]);
        input.ids.front() = 42;
        const auto mutated = std::make_shared<const ValidatedDataset>(input.columns, input.embeddings, input.ids);
        const Request substitution{mutated, work.queries, predicates, top_k, batch_size, query_count, Consent::accept_g, false};
        rejected("changed_data_same_shape", substitution, Reason::identity_mismatch);
    }
    {
        const std::array<RecordID, 1> ids{1};
        const ColumnBatch cols({std::vector<std::int32_t>{7}});
        const EmbeddingBuffer few(1, dimension, std::vector<float>(dimension, 1.0f));
        const auto fewer = std::make_shared<const ValidatedDataset>(cols, few, ids);
        const Request wrong_n{fewer, work.queries, predicates, top_k, batch_size, query_count, Consent::accept_g, false};
        rejected("changed_n", wrong_n, Reason::evidence_mismatch);
        std::vector<RecordID> distinct(100'000);
        for (std::size_t i = 0; i < distinct.size(); ++i) { distinct[i] = i; }
        const ColumnBatch matching({std::vector<std::int32_t>(distinct.size(), 7)});
        const EmbeddingBuffer narrow(distinct.size(), 1, std::vector<float>(distinct.size(), 1.0f));
        const auto different_d = std::make_shared<const ValidatedDataset>(matching, narrow, distinct);
        const Request wrong_d{different_d, work.queries, predicates, top_k, batch_size, query_count, Consent::accept_g, false};
        rejected("changed_d", wrong_d, Reason::evidence_mismatch);
    }
    require(std::fesetround(FE_DOWNWARD) == 0, "cannot challenge placement FP settings");
    rejected("wrong_fp_environment", accepted, Reason::device_context_unsupported);
    require(std::fesetround(FE_TONEAREST) == 0, "cannot restore experimental rounding");
}

void gpu_tests(const Golden& golden, const char* raw_path) {
    const auto work = Workload::historical(true);
    const auto request = request_for(work, Consent::accept_g);
    ResidentHandle resident(work, work.snapshot->columns().row_count());
    gpu_decision_tests(work, resident);
    {
        std::array<cuda::CandidateEvidence, query_count> evidence;
        const auto first = run_session(work, request, &resident, raw_path ? &evidence : nullptr);
        check_decision("gpu_executed", first.decision, {Backend::gpu, Reason::verified_evidence});
        verify_top(first, golden);
        const auto repeated = run_session(work, request, &resident);
        check_decision("gpu_reused", repeated.decision, {Backend::gpu, Reason::verified_evidence});
        verify_top(repeated, golden);
        resident.close();
        // The saved output and candidate records own their data after close.
        verify_top(first, golden);
        if (raw_path) { verify_candidates(work, golden, raw_path, evidence); }
    }
    check_decision("closed", choose(work, request, &resident), {Backend::none, Reason::closed_or_invalidated, true});
    ResidentHandle insufficient(work, cells[3].matches - 1);
    check_decision("insufficient_capacity", choose(work, request, &insufficient), {Backend::none, Reason::capacity_insufficient, true});
    insufficient.close();
    std::cout << "PASS gpu placement\n";
}

void fault_test() {
    const auto work = Workload::historical(true);
    auto request = request_for(work, Consent::accept_g, true);
    ResidentHandle resident(work, work.snapshot->columns().row_count());
    check_decision("before_injected_failure", choose(work, request, &resident), {Backend::gpu, Reason::verified_evidence});
    try {
        (void)run_session(work, request, &resident);
        throw std::logic_error("injected CUDA error produced output");
    } catch (const std::runtime_error& error) {
        require(std::string_view(error.what()).find("kernel submit") != std::string_view::npos,
                "fault did not occur after GPU query began");
        std::cout << "reported CUDA failure without results\n";
    }
    request.allow_cpu = false;
    check_decision("invalidated_after_failure", choose(work, request, &resident),
                   {Backend::none, Reason::closed_or_invalidated, true});
    resident.close();
    std::cout << "PASS fault placement\n";
}

void context_test() {
    const auto work = Workload::historical(true);
    ResidentHandle resident(work, work.snapshot->columns().row_count());
    check_decision("injected_context_mismatch", choose(work, request_for(work, Consent::accept_g), &resident),
                   {Backend::none, Reason::device_context_unsupported, true});
    resident.close();
    std::cout << "PASS context placement\n";
}

void measure_placement(const Golden& golden, bool gpu) {
    const auto work = Workload::historical(true);
    const auto request = request_for(work, gpu ? Consent::accept_g : Consent::strict_s);
    std::optional<ResidentHandle> resident;
    if (gpu) { resident.emplace(work, work.snapshot->columns().row_count()); }
    const auto result = run_session(work, request, resident ? &*resident : nullptr);
    require(result.decision == (gpu ? Decision{Backend::gpu, Reason::verified_evidence} :
                                     Decision{Backend::cpu, Reason::strict_cpu}), "wrong measured backend");
    verify_top(result, golden);
    if (resident) { resident->close(); }
    emit_measurement(result, resident ? resident->preparation() : cuda::Preparation{});
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 2, "placement requires a mode");
        const std::string_view mode = argv[1];
        if (mode == "--cpu-tests" && argc == 2) { cpu_tests(); return 0; }
        // These test/measurement modes explicitly select the historical FP domain.
        require(argc >= 3, "placement mode requires historical S90 expectations");
        const NearestEnvironment environment;
        require(common_fp_domain(), "placement fixture requires the common FP domain");
        const auto golden = read_golden(argv[2], cells[3]);
        require(golden.data_hash == s90_data && golden.query_hash == historical_queries, "wrong placement evidence");
        if (mode == "--gpu-tests" && (argc == 3 || argc == 4)) {
            gpu_tests(golden, argc == 4 ? argv[3] : nullptr);
        } else if (mode == "--fault-test" && argc == 3) {
            fault_test();
        } else if (mode == "--context-test" && argc == 3) {
            context_test();
        } else if (mode == "--measure" && argc == 4) {
            const std::string_view path = argv[3];
            require(path == "cpu" || path == "gpu", "unknown measurement path");
            measure_placement(golden, path == "gpu");
        } else {
            throw std::invalid_argument("invalid placement arguments");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "placement_failed: " << error.what() << '\n';
        return 1;
    }
}
