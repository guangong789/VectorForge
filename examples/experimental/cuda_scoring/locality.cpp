#define VECTORFORGE_PRIVATE_RETRIEVAL_REUSE
#include "retrieval.cpp"

#include <iomanip>
#include <numeric>

namespace {

constexpr Cell locality_cell{"V2-01", 100'000, 10'000};
constexpr std::uint64_t scatter_seed = 0x563230314C4F43ULL;

enum class Layout { clustered, scattered };
enum class Trial { replay, replay_events, cpu, gpu, gpu_phases, replay_warmed };

struct StudyFixture {
    Queries queries = query_inputs();
    std::unique_ptr<const ValidatedDataset> snapshot;
    // Both layouts retain the same logical ID order in the actual physical scan.
    std::vector<gpu::RowIndex> rows;
    std::vector<RecordID> selected_ids;
    Expected expected;
    std::array<std::vector<gpu::Score>, query_count> scores;
    std::uint64_t logical = 0, physical = 0, indices = 0, order = 0, membership = 0;
    std::size_t min_gap = 0, max_gap = 0;
};

std::uint32_t bits(float x) { return std::bit_cast<std::uint32_t>(x); }

void add_metadata(Fingerprint& hash, const Metadata& metadata) {
    hash.add(metadata.id);
    hash.add(static_cast<std::uint32_t>(metadata.tenant_id));
    hash.add(static_cast<std::uint32_t>(metadata.category_id));
    hash.add(static_cast<std::uint64_t>(metadata.updated_at));
    hash.add(static_cast<std::uint32_t>(metadata.token_count));
}

StudyFixture make_fixture(Layout layout) {
    StudyFixture fixture;
    const auto logical = generate(locality_cell);
    fixture.expected = independent_oracle(logical, fixture.queries, locality_cell);
    std::vector<std::size_t> selected, rejected;
    Fingerprint logical_hash;
    for (std::size_t row = 0; row < logical.rows.size(); ++row) {
        const auto& record = logical.rows[row];
        add_metadata(logical_hash, record);
        (matches(record) ? selected : rejected).push_back(row);
    }
    for (float f : logical.embeddings) { logical_hash.add(bits(f)); }
    fixture.logical = logical_hash.value;
    require(selected.size() == locality_cell.matches, "incorrect logical membership");
    for (std::size_t j = 0; j < query_count; ++j) {
        for (const auto row : selected) {
            const auto value = oracle_score(std::span(logical.embeddings).subspan(row * dimension, dimension),
                                            fixture.queries[j]);
            require(value.first_failure == gpu::no_failure, "benchmark input overflowed");
            fixture.scores[j].push_back(value);
        }
    }

    std::vector<gpu::RowIndex> slots(locality_cell.rows);
    std::iota(slots.begin(), slots.end(), gpu::RowIndex{0});
    if (layout == Layout::scattered) {
        Random rng(scatter_seed);
        for (std::size_t remaining = slots.size(); remaining > 1; --remaining) {
            std::swap(slots[remaining - 1], slots[rng.index(remaining)]);
        }
    }
    slots.resize(selected.size());
    std::sort(slots.begin(), slots.end());
    require(std::adjacent_find(slots.begin(), slots.end()) == slots.end(), "duplicate physical slot");
    fixture.rows = std::move(slots);

    LogicalData physical{std::vector<Metadata>(locality_cell.rows),
                         std::vector<float>(locality_cell.rows * dimension)};
    std::vector<std::size_t> physical_to_logical(locality_cell.rows);
    std::size_t included = 0, excluded = 0;
    for (std::size_t row = 0; row < physical.rows.size(); ++row) {
        const bool hit = included < fixture.rows.size() && fixture.rows[included] == row;
        const auto source = hit ? selected.at(included++) : rejected.at(excluded++);
        physical_to_logical[row] = source;
        physical.rows[row] = logical.rows[source];
        std::copy_n(logical.embeddings.data() + source * dimension, dimension,
                    physical.embeddings.data() + row * dimension);
    }
    require(included == selected.size() && excluded == rejected.size(), "layout lost logical records");
    {
        const auto inputs = own_inputs(std::move(physical));
        fixture.physical = input_fingerprint(inputs);
        fixture.snapshot = std::make_unique<const ValidatedDataset>(inputs.columns, inputs.embeddings, inputs.ids);
    }
    const auto& cols = fixture.snapshot->columns();
    const auto tenants = cols.column<std::int32_t>(tenant_column);
    const auto categories = cols.column<std::int32_t>(category_column);
    const auto timestamps = cols.column<std::int64_t>(updated_column);
    const auto tokens = cols.column<std::int32_t>(tokens_column);
    std::vector<bool> seen(locality_cell.rows);
    Fingerprint indices, order, membership;
    fixture.min_gap = locality_cell.rows;
    std::size_t matched = 0;
    for (std::size_t r = 0; r < physical_to_logical.size(); ++r) {
        const auto i = physical_to_logical[r];
        require(!seen.at(i), "not a permutation");
        seen[i] = true;
        const auto& ref = logical.rows[i];
        const auto id = fixture.snapshot->record_ids()[r];
        require(id == ref.id && tenants[r] == ref.tenant_id && categories[r] == ref.category_id &&
                timestamps[r] == ref.updated_at && tokens[r] == ref.token_count, "metadata or ID changed");
        const auto expected_row = std::span(logical.embeddings).subspan(i * dimension, dimension);
        const auto admitted_row = fixture.snapshot->embeddings().row(r);
        require(std::ranges::equal(expected_row, admitted_row,
                [](float a, float b) { return bits(a) == bits(b); }), "embedding bits changed");
        if (!matches({id, timestamps[r], tenants[r], categories[r], tokens[r]})) { continue; }
        require(matched < selected.size() && fixture.rows[matched] == r && selected[matched] == i,
                "physical scan changed candidate membership or logical order");
        if (matched) {
            const auto gap = r - fixture.rows[matched - 1];
            fixture.min_gap = std::min(fixture.min_gap, gap);
            fixture.max_gap = std::max(fixture.max_gap, gap);
        }
        ++matched;
        fixture.selected_ids.push_back(id);
        order.add(id);
        indices.add(r);
    }
    require(matched == selected.size() && count_where(*fixture.snapshot, predicates, batch_size) == selected.size(),
            "actual candidate count changed");
    require(std::ranges::is_sorted(fixture.selected_ids) &&
            std::adjacent_find(fixture.selected_ids.begin(), fixture.selected_ids.end()) == fixture.selected_ids.end(),
            "slots not in strict RecordID order");
    for (const auto i : selected) { membership.add(logical.rows[i].id); }
    fixture.order = order.value;
    fixture.indices = indices.value;
    fixture.membership = membership.value;
    require(fixture.order == fixture.membership, "logical slot fingerprint changed");
    return fixture;
}

void check_scores(std::span<const gpu::Score> actual, std::span<const gpu::Score> expected) {
    require(actual.size() == expected.size(), "score count changed");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(actual[i].first_failure == expected[i].first_failure &&
                bits(actual[i].value) == bits(expected[i].value), "candidate score/status bit mismatch");
    }
}

void identity(const StudyFixture& f, std::string_view layout) {
    std::cout << "{\"kind\":\"identity\",\"layout\":\"" << layout << "\",\"logical\":" << f.logical
              << ",\"physical\":" << f.physical << ",\"queries\":" << query_fingerprint(f.queries)
              << ",\"logical_order\":" << f.order << ",\"selected_ids\":" << f.membership
              << ",\"physical_indices\":" << f.indices << ",\"candidate_count\":" << f.rows.size()
              << ",\"first_row\":" << f.rows.front() << ",\"last_row\":" << f.rows.back()
              << ",\"min_gap\":" << f.min_gap << ",\"max_gap\":" << f.max_gap << ",\"top\":[";
    for (std::size_t j = 0; j < query_count; ++j) {
        std::cout << (j ? "," : "") << result_fingerprint(f.expected.ranked[j]);
    }
    std::cout << "],\"scores\":[";
    for (std::size_t j = 0; j < query_count; ++j) {
        Fingerprint hash;
        for (const auto& score : f.scores[j]) {
            hash.add(bits(score.value));
            hash.add(score.first_failure);
        }
        std::cout << (j ? "," : "") << hash.value;
    }
    std::cout << "]}\n";
}

#ifndef VECTORFORGE_PRIVATE_LOCALITY_REUSE
// The same immutable snapshot supplies both the staging upload and retrieval.
class LocalResident {
public:
    LocalResident(const ValidatedDataset& snapshot, bool events)
        : snapshot_(snapshot) {
        const auto start = now_ns();
        gpu::select_device(false);
        const auto context_complete = now_ns();
        context_ns = context_complete - start;
        {
            const auto flat = pack(snapshot_);
            pack_ns = now_ns() - context_complete;
            create_resident(scorer_, snapshot_, flat, memory);
        }
        resident_ns = now_ns() - context_complete - pack_ns;
        if (events) {
            const auto event_start = now_ns();
            scorer_->enable_device_timing();
            event_setup_ns = now_ns() - event_start;
        }
    }

    std::vector<gpu::Score> replay(std::span<const gpu::RowIndex> rows, std::span<const float> q,
                                  gpu::ScorePhases* host = nullptr, gpu::DevicePhases* device = nullptr) {
        return scorer_->score(rows, q, host, device);
    }

    std::vector<ScoredRecord> retrieve(std::span<const float> q, std::span<const Predicate> filters,
                                       std::size_t k, Diagnostic* phases = nullptr, CandidateTrace* trace = nullptr) {
        return retrieve_gpu(snapshot_, *scorer_, q, filters, k, batch_size, phases, trace);
    }

    void close() { scorer_->close(); }
    gpu::ResidencyStats memory;
    Tick context_ns = 0, pack_ns = 0, resident_ns = 0, event_setup_ns = 0;

private:
    const ValidatedDataset& snapshot_;
    std::optional<gpu::ResidentScorer> scorer_;
};

void check_cpu(const StudyFixture& f) {
    for (std::size_t j = 0; j < query_count; ++j) {
        exact(execute_batched(*f.snapshot, f.queries[j], predicates, top_k, batch_size), f.expected.ranked[j]);
        exact(execute_scalar_reference(*f.snapshot, f.queries[j], predicates, top_k), f.expected.ranked[j]);
    }
}

void check_event_reading(const gpu::DevicePhases& d, Tick wall) {
    require(d.h2d_ms > 0 && d.kernel_ms > 0 && d.d2h_ms > 0 && d.read_ns > 0 &&
            (static_cast<double>(d.h2d_ms) + d.kernel_ms + d.d2h_ms) * 1e6 <= static_cast<double>(wall) + 2e3,
            "invalid completed CUDA event intervals");
}

void check_gpu(const StudyFixture& f) {
    LocalResident resident(*f.snapshot, true);
    for (std::size_t j = 0; j < query_count; ++j) {
        CandidateTrace trace;
        exact(resident.retrieve(f.queries[j], predicates, top_k, nullptr, &trace), f.expected.ranked[j]);
        require(trace.rows == f.rows, "filter candidate order changed");
        check_scores(trace.scores, f.scores[j]);
        gpu::DevicePhases elapsed;
        const auto start = now_ns();
        check_scores(resident.replay(f.rows, f.queries[j], nullptr, &elapsed), f.scores[j]);
        check_event_reading(elapsed, now_ns() - start);
    }
    gpu::DevicePhases empty;
    empty.kernel_ms = 7;
    require(resident.replay({}, f.queries[0], nullptr, &empty).empty() && empty.h2d_ms == 0 &&
            empty.kernel_ms == 0 && empty.d2h_ms == 0 && empty.read_ns == 0, "empty replay enqueued device work");
    const std::array no_match{Predicate{tenant_column, Comparison::equal, std::int32_t{-1}}};
    require(resident.retrieve(f.queries[0], no_match, top_k).empty(), "empty retrieval produced a result");
    const std::array invalid_rows{static_cast<gpu::RowIndex>(locality_cell.rows)};
    reject_input<std::out_of_range>("locality bad index", [&] { (void)resident.replay(invalid_rows, f.queries[0]); });
    auto invalid_query = f.queries[0];
    invalid_query[0] = std::numeric_limits<float>::infinity();
    reject_input<std::invalid_argument>("locality nonfinite input",
                                        [&] { (void)resident.replay(f.rows, invalid_query); });
    resident.close();
    reject_input<std::logic_error>("locality closed event owner",
                                    [&] { (void)resident.replay(f.rows, f.queries[0], nullptr, &empty); });

    const float max = std::numeric_limits<float>::max();
    const EmbeddingBuffer bad_embeddings(3, 2, {1.0f, 0.0f, max / 2, max, max, 0.0f});
    const ColumnBatch bad_cols({std::vector<std::int32_t>{1, 1, 1}});
    const std::array<RecordID, 3> ids{9, 3, 7};
    const ValidatedDataset bad_snapshot(bad_cols, bad_embeddings, ids);
    LocalResident bad(bad_snapshot, true);
    const std::array query{2.0f, 1.0f};
    const std::array<gpu::RowIndex, 3> rows{0, 1, 2};
    const std::array<gpu::Score, 3> want{{{2.0f, gpu::no_failure}, {0.0f, 1}, {0.0f, 0}}};
    gpu::DevicePhases overflow_timing;
    const auto start = now_ns();
    const auto scores = bad.replay(rows, query, nullptr, &overflow_timing);
    check_event_reading(overflow_timing, now_ns() - start);
    check_scores(scores, want);
    try {
        (void)bad.retrieve(query, {}, 1);
        throw std::runtime_error("failure path published a result");
    } catch (const ScoreFailure& error) {
        require(error.slot == 1 && error.row == 1 && error.feature == 1, "first failure ordering changed");
    }
    bad.close();
    std::cout << "{\"kind\":\"gpu_gate\",\"candidate_checks\":160000,\"top_k_queries\":8,"
                 "\"event_checks\":\"completed/empty/invalid/overflow/closed\",\"pass\":true}\n";
}

struct Sample {
    Tick query_ns = 0;
    Diagnostic phases;
    gpu::DevicePhases device;
};

void emit_sample(std::size_t query, const Sample& sample, std::string_view kind = "query") {
    const auto& p = sample.phases;
    const auto& h = p.scoring;
    const auto& d = sample.device;
    std::cout << "{\"kind\":\"" << kind << "\",\"query\":" << query << ",\"wall_ns\":" << sample.query_ns
              << ",\"validate_ns\":" << p.validate << ",\"reserve_ns\":" << p.row_reserve
              << ",\"filter_emit_ns\":" << p.scan_emit << ",\"rank_ns\":" << p.rank
              << ",\"prepare_ns\":" << h.prepare_ns << ",\"unpack_ns\":" << h.unpack_ns
              << ",\"h2d_submit_ns\":" << h.h2d_submit_ns << ",\"h2d_wait_ns\":" << h.h2d_wait_ns
              << ",\"kernel_submit_ns\":" << h.kernel_submit_ns << ",\"kernel_wait_ns\":" << h.kernel_wait_ns
              << ",\"d2h_submit_ns\":" << h.d2h_submit_ns << ",\"d2h_wait_ns\":" << h.d2h_wait_ns
              << ",\"event_h2d_ms\":" << d.h2d_ms << ",\"event_kernel_ms\":" << d.kernel_ms
              << ",\"event_d2h_ms\":" << d.d2h_ms << ",\"event_read_ns\":" << d.read_ns << "}\n";
}

void run_trial(const StudyFixture& f, Trial trial, std::string_view mode, Tick dataset_ns) {
    const bool device = trial != Trial::cpu;
    const bool warmed = trial == Trial::replay_warmed;
    const bool events = trial == Trial::replay_events || warmed;
    const bool replay = trial == Trial::replay || events;
    std::optional<LocalResident> resident;
    if (device) { resident.emplace(*f.snapshot, events); }
    std::array<Sample, query_count> samples{};
    std::array<Top, query_count> consumed{};
    std::array<std::vector<gpu::Score>, query_count> replay_results;
    constexpr std::size_t prime_query = 0;
    constexpr std::array<std::string_view, 2> prefix_kinds{"first_query", "warmup"};
    std::array<Sample, prefix_kinds.size()> prefix{};
    if (warmed) {
        // Observe the first score call and one explicit prime on the same owner.
        // Both bit checks finish before the eight-query timed session.
        for (auto& sample : prefix) {
            const auto start = now_ns();
            const auto scores = resident->replay(f.rows, f.queries[prime_query], &sample.phases.scoring,
                                                 &sample.device);
            sample.query_ns = now_ns() - start;
            check_scores(scores, f.scores[prime_query]);
            check_event_reading(sample.device, sample.query_ns);
        }
    }
    const auto begin = now_ns();
    const auto selected = resident && !replay; // Explicit experimental routing. No historical placement rule.
    for (std::size_t j = 0; j < query_count; ++j) {
        auto& sample = samples[j];
        const auto start = now_ns();
        if (replay) {
            replay_results[j] = resident->replay(f.rows, f.queries[j], events ? &sample.phases.scoring : nullptr,
                                                  events ? &sample.device : nullptr);
        } else {
            auto owned = selected ? resident->retrieve(f.queries[j], predicates, top_k,
                                                        trial == Trial::gpu_phases ? &sample.phases : nullptr)
                                  : execute_batched(*f.snapshot, f.queries[j], predicates, top_k, batch_size);
            require(owned.size() == top_k, "timed result count changed");
            std::copy(owned.begin(), owned.end(), consumed[j].begin());
        }
        sample.query_ns = now_ns() - start;
    }
    const auto duration = now_ns() - begin;
    for (std::size_t j = 0; j < query_count; ++j) {
        if (replay) { check_scores(replay_results[j], f.scores[j]); }
        else { exact(consumed[j], f.expected.ranked[j]); }
        if (events) { check_event_reading(samples[j].device, samples[j].query_ns); }
    }
    const auto close_start = now_ns();
    if (resident) { resident->close(); }
    const auto close_ns = now_ns() - close_start;
    const auto memory = resident ? resident->memory : gpu::ResidencyStats{};
    std::cout << "{\"kind\":\"trial\",\"mode\":\"" << mode << "\",\"session_ns\":" << duration
              << ",\"fixture_ns\":" << dataset_ns << ",\"context_ns\":" << (resident ? resident->context_ns : 0)
              << ",\"pack_ns\":" << (resident ? resident->pack_ns : 0)
              << ",\"residency_ns\":" << (resident ? resident->resident_ns : 0)
              << ",\"event_setup_ns\":" << (resident ? resident->event_setup_ns : 0)
              << ",\"allocation_ns\":" << memory.allocation_ns << ",\"upload_ns\":" << memory.upload_ns
              << ",\"device_bytes\":" << memory.device_bytes << ",\"pinned_bytes\":" << memory.pinned_bytes
              << ",\"initial_free_device_bytes\":" << memory.free_device_bytes << ",\"close_ns\":" << (resident ? close_ns : 0)
              << ",\"scores_checked\":" << (replay ? query_count * locality_cell.matches : 0)
              << ",\"prefix_scores_checked\":" << (warmed ? prefix.size() * locality_cell.matches : 0)
              << ",\"top_k_queries\":" << (replay ? 0 : query_count) << ",\"pass\":true}\n";
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    if (warmed) {
        for (std::size_t i = 0; i < prefix.size(); ++i) { emit_sample(prime_query, prefix[i], prefix_kinds[i]); }
    }
    for (std::size_t j = 0; j < query_count; ++j) { emit_sample(j, samples[j]); }
}
#endif

} // namespace

#ifndef VECTORFORGE_PRIVATE_LOCALITY_REUSE
int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: vectorforge_cuda_locality MODE clustered|scattered");
        const NearestEnvironment environment;
#if defined(__x86_64__)
        require((_mm_getcsr() & 0x8040U) == 0, "host FTZ/DAZ must be off");
#endif
        const std::string_view mode = argv[1], layout = argv[2];
        require(layout == "clustered" || layout == "scattered", "unknown layout");
        const auto start = now_ns();
        const auto f = make_fixture(layout == "clustered" ? Layout::clustered : Layout::scattered);
        const auto fixture_ns = now_ns() - start;
        identity(f, layout);
        if (mode == "--host-gate") {
            check_cpu(f);
            std::cout << "{\"kind\":\"host_gate\",\"top_k_queries\":8,\"pass\":true}\n";
        } else if (mode == "--gate") {
            check_cpu(f);
            check_gpu(f);
        } else {
            constexpr std::array<std::string_view, 6> modes{
                "--replay", "--replay-events", "--cpu", "--gpu", "--gpu-phases", "--replay-warmed"};
            const auto found = std::ranges::find(modes, mode);
            require(found != modes.end(), "unknown study mode");
            run_trial(f, static_cast<Trial>(found - modes.begin()), mode, fixture_ns);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "locality_failed: " << error.what() << '\n';
        return 1;
    }
}
#endif
