#define VECTORFORGE_PRIVATE_LOCALITY_REUSE
#include "locality.cpp"

namespace {

struct QuerySample {
    gpu::ScorePhases host;
    gpu::DevicePhases device;
    Tick wall_ns = 0;
    Tick results_ns = 0;
};

std::vector<ScoredRecord> rank_slots(std::span<const gpu::Score> scores,
                                     std::span<const gpu::RowIndex> rows,
                                     std::span<const RecordID> source_ids,
                                     std::span<const RecordID> compact_ids) {
    require(scores.size() == rows.size(), "GPU score count changed");
    for (std::size_t slot = 0; slot < scores.size(); ++slot) {
        if (scores[slot].first_failure != gpu::no_failure) {
            throw ScoreFailure(slot, rows[slot], scores[slot].first_failure);
        }
        require(std::isfinite(scores[slot].value), "non-finite successful GPU score");
    }

    std::vector<ScoredRecord> records;
    records.reserve(rows.size());
    for (std::size_t slot = 0; slot < rows.size(); ++slot) {
        const auto id = compact_ids.empty() ? source_ids[rows[slot]] : compact_ids[slot];
        records.push_back({id, scores[slot].value});
    }
    const auto limit = std::min(top_k, records.size());
    std::partial_sort(records.begin(), records.begin() + static_cast<std::ptrdiff_t>(limit), records.end(),
                      [](const auto& a, const auto& b) {
                          return a.score > b.score || (a.score == b.score && a.id < b.id);
                      });
    records.resize(limit);
    return records;
}

class FixedCandidateSession {
public:
    explicit FixedCandidateSession(const StudyFixture& f) : fixture(f) {
        const auto start = now_ns();
        gpu::select_device(false);
        const auto selected = now_ns();
        const auto flat = pack(*fixture.snapshot);
        const auto packed = now_ns();
        create_resident(scorer, *fixture.snapshot, flat, memory);
        const auto resident = now_ns();
        scorer->enable_device_timing();
        context_ns = selected - start;
        pack_ns = packed - selected;
        residency_ns = resident - packed;
        event_setup_ns = now_ns() - resident;
    }

    void prepare(gpu::PreparationPhases& phases) {
        scorer->prepare_candidates(fixture.rows, phases);
        const auto start = now_ns();
        ids.reserve(fixture.rows.size());
        const auto source = fixture.snapshot->record_ids();
        for (const auto row : fixture.rows) { ids.push_back(source[row]); }
        mapping_ns = now_ns() - start;
    }

    void query(bool prepared, std::size_t query_index, Top& top, QuerySample& sample) {
        const auto start = now_ns();
        Tick rank_start = 0;
        {
            const auto q = std::span(fixture.queries[query_index]);
            const auto scores = prepared ? scorer->score_prepared(q, &sample.host, &sample.device)
                                         : scorer->score(fixture.rows, q, &sample.host, &sample.device);
            rank_start = now_ns();
            const auto top_k_results = rank_slots(scores, fixture.rows, fixture.snapshot->record_ids(), ids);
            require(top_k_results.size() == top.size(), "query returned unexpected Top-K size");
            std::ranges::copy(top_k_results, top.begin());
        }
        const auto end = now_ns();
        sample.wall_ns = end - start;
        sample.results_ns = end - rank_start;
    }

    void release_prepared() {
        scorer->release_prepared();
        std::vector<RecordID>{}.swap(ids);
    }

    const StudyFixture& fixture;
    gpu::ResidencyStats memory;
    Tick context_ns = 0, pack_ns = 0, residency_ns = 0, event_setup_ns = 0, mapping_ns = 0;
    std::vector<RecordID> ids;
    std::optional<gpu::ResidentScorer> scorer;
};

void check_prepared(const StudyFixture& f) {
    FixedCandidateSession run(f);
    gpu::PreparationPhases phases;
    run.prepare(phases);
    require(run.ids == f.selected_ids, "compact slot RecordIDs differ");
    require(phases.additional_device_bytes == f.rows.size() * (dimension * sizeof(float) + sizeof(gpu::RowIndex)),
            "unexpected aggregate compact allocation");
    const auto copies = run.scorer->read_prepared_for_check();
    require(copies.size() == f.rows.size() * dimension, "prepared embedding size changed");
    for (std::size_t slot = 0; slot < f.rows.size(); ++slot) {
        require(std::ranges::equal(f.snapshot->embeddings().row(f.rows[slot]),
                std::span(copies).subspan(slot * dimension, dimension),
                [](float a, float b) { return bits(a) == bits(b); }), "prepared embedding bits differ");
    }

    for (std::size_t q = 0; q < query_count; ++q) {
        const auto direct = run.scorer->score(f.rows, f.queries[q]);
        const auto prepared = run.scorer->score_prepared(f.queries[q]);
        check_scores(direct, f.scores[q]);
        check_scores(prepared, f.scores[q]);
        const auto ids = f.snapshot->record_ids();
        exact(rank_slots(direct, f.rows, ids, {}), f.expected.ranked[q]);
        exact(rank_slots(prepared, f.rows, ids, run.ids), f.expected.ranked[q]);
        exact(execute_scalar_reference(*f.snapshot, f.queries[q], predicates, top_k), f.expected.ranked[q]);
        exact(execute_batched(*f.snapshot, f.queries[q], predicates, top_k, batch_size), f.expected.ranked[q]);
    }
    reject_input<std::logic_error>("different candidate sequence on the same owner", [&] {
        run.scorer->prepare_candidates(std::span(f.rows).first(1), phases);
    });
    reject_input<std::invalid_argument>("wrong query dimension", [&] {
        (void)run.scorer->score_prepared(std::span(f.queries[0]).first(dimension - 1));
    });
    auto invalid = f.queries[0];
    invalid[0] = std::numeric_limits<float>::quiet_NaN();
    reject_input<std::invalid_argument>("nonfinite query", [&] { (void)run.scorer->score_prepared(invalid); });
    run.release_prepared();
    run.release_prepared();
    reject_input<std::logic_error>("prepared scoring after release", [&] {
        (void)run.scorer->score_prepared(f.queries[0]);
    });
    check_scores(run.scorer->score(f.rows, f.queries[0]), f.scores[0]);
    run.scorer->close();
    run.scorer->close();
    reject_input<std::logic_error>("closed owner", [&] { (void)run.scorer->score_prepared(f.queries[0]); });

    const auto max = std::numeric_limits<float>::max();
    const std::array inputs{1.0f, -0.0f, max, 0.0f};
    const std::array q{2.0f, 1.0f};
    gpu::ResidentScorer empty(inputs, 2, 2, 1, false);
    empty.enable_device_timing();
    const std::array<gpu::RowIndex, 1> invalid_rows{2};
    const std::array<gpu::RowIndex, 2> over_capacity{0, 1};
    reject_input<std::out_of_range>("invalid candidate index", [&] { empty.prepare_candidates(invalid_rows, phases); });
    reject_input<std::invalid_argument>("candidate capacity", [&] { empty.prepare_candidates(over_capacity, phases); });
    empty.prepare_candidates({}, phases);
    require(empty.score_prepared(q).empty() && empty.read_prepared_for_check().empty() &&
            phases.additional_device_bytes == 0, "empty candidates allocated or scored");
    empty.close();

    gpu::ResidentScorer single(inputs, 2, 2, 1, false);
    single.enable_device_timing();
    const std::array<gpu::RowIndex, 1> one{0};
    single.prepare_candidates(one, phases);
    const std::array<gpu::Score, 1> good{{{2.0f, gpu::no_failure}}};
    check_scores(single.score_prepared(q), good);
    require(bits(single.read_prepared_for_check()[1]) == bits(-0.0f), "gather changed negative zero");
    single.close();

    gpu::ResidentScorer failure(inputs, 2, 2, 2, false);
    failure.enable_device_timing();
    const std::array<gpu::RowIndex, 2> overflow_rows{1, 0};
    const std::array<RecordID, 2> overflow_ids{71, 12};
    failure.prepare_candidates(overflow_rows, phases);
    const auto values = failure.score_prepared(q);
    const std::array<gpu::Score, 2> want{{{0.0f, 0}, {2.0f, gpu::no_failure}}};
    check_scores(values, want);
    check_scores(failure.score(overflow_rows, q), want);
    try {
        (void)rank_slots(values, overflow_rows, overflow_ids, {});
        throw std::runtime_error("failed query produced Top-K");
    } catch (const ScoreFailure& error) {
        require(error.slot == 0 && error.row == 1 && error.feature == 0, "failure mapping changed");
    }
    failure.close();
    std::cout << "{\"kind\":\"gate\",\"pass\":true,\"embedding_bits\":1280000,"
                 "\"direct_score_status\":80000,\"prepared_score_status\":80000,\"top_k\":16}\n";
}

void check_budget() {
    constexpr std::size_t rows = 250'000, candidate_capacity = 12'000;
    const std::vector<float> values(rows * dimension, 0.0f);
    const std::vector<gpu::RowIndex> selected(candidate_capacity, 0);
    gpu::select_device(false);
    gpu::ResidentScorer scorer(values, rows, dimension, candidate_capacity, false);
    scorer.enable_device_timing();
    gpu::PreparationPhases phases;
    reject_input<std::length_error>("aggregate device budget", [&] { scorer.prepare_candidates(selected, phases); });
    require(phases.additional_device_bytes == 0, "over-cap allocation recorded");
    scorer.close();
}

constexpr std::array<std::size_t, 5> horizons{1, 8, 32, 64, 128};

void trial(const StudyFixture& f, std::size_t count, bool prepared) {
    FixedCandidateSession run(f);
    const auto host_start = now_ns();
    gpu::PreparationPhases prep;
    Tick preparation_ns = 0;
    if (prepared) {
        run.prepare(prep);
        preparation_ns = now_ns() - host_start;
    }
    std::array<Top, horizons.back()> tops{};
    std::array<QuerySample, horizons.back()> samples{};
    const auto session_start = now_ns();
    for (std::size_t i = 0; i < count; ++i) {
        run.query(prepared, i % query_count, tops[i], samples[i]);
    }
    const auto session_ns = now_ns() - session_start;
    Tick release_ns = 0;
    if (prepared) {
        const auto close = now_ns();
        run.release_prepared();
        release_ns = now_ns() - close;
    }
    const auto common_release_start = now_ns();
    run.scorer->close();
    const auto common_release_ns = now_ns() - common_release_start;
    for (std::size_t i = 0; i < count; ++i) {
        exact(tops[i], f.expected.ranked[i % query_count]);
        const auto& d = samples[i].device;
        require(d.h2d_ms > 0 && d.kernel_ms > 0 && d.d2h_ms > 0, "unusable query event interval");
    }
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
              << "{\"kind\":\"session\",\"strategy\":\"" << (prepared ? "prepared" : "direct")
              << "\",\"q\":" << count << ",\"context_ns\":" << run.context_ns << ",\"pack_ns\":" << run.pack_ns
              << ",\"resident_ns\":" << run.residency_ns << ",\"event_setup_ns\":" << run.event_setup_ns
              << ",\"base_allocation_ns\":" << run.memory.allocation_ns << ",\"base_upload_ns\":" << run.memory.upload_ns
              << ",\"base_free_bytes\":" << run.memory.free_device_bytes << ",\"base_device_bytes\":" << run.memory.device_bytes
              << ",\"pinned_bytes\":" << run.memory.pinned_bytes << ",\"compact_device_bytes\":" << prep.additional_device_bytes
              << ",\"compact_host_mapping_bytes\":" << (prepared ? f.rows.size() * sizeof(RecordID) : 0)
              << ",\"prep_free_bytes\":" << prep.free_device_bytes << ",\"common_cleanup_ns\":" << common_release_ns
              << ",\"preparation_ns\":" << preparation_ns << ",\"session_ns\":" << session_ns
              << ",\"cleanup_ns\":" << release_ns << ",\"execution_ns\":" << preparation_ns + session_ns
              << ",\"inclusive_ns\":" << preparation_ns + session_ns + release_ns << ",\"pass\":true}\n";
    std::cout << "{\"kind\":\"preparation\",\"validation_ns\":" << prep.validation_ns
              << ",\"allocation_ns\":" << prep.allocation_ns << ",\"staging_ns\":" << prep.staging_ns
              << ",\"index_submit_ns\":" << prep.h2d_submit_ns << ",\"index_wait_ns\":" << prep.h2d_wait_ns
              << ",\"gather_submit_ns\":" << prep.gather_submit_ns << ",\"gather_wait_ns\":" << prep.gather_wait_ns
              << ",\"index_h2d_ms\":" << prep.h2d_ms << ",\"gather_ms\":" << prep.gather_ms
              << ",\"event_read_ns\":" << prep.event_read_ns << ",\"mapping_ns\":" << run.mapping_ns << "}\n";
    for (std::size_t i = 0; i < count; ++i) {
        const auto& s = samples[i];
        const auto& h = s.host;
        const auto& d = s.device;
        std::cout << "{\"kind\":\"query\",\"i\":" << i << ",\"vector\":" << i % query_count
                  << ",\"top_fingerprint\":" << result_fingerprint(tops[i])
                  << ",\"wall_ns\":" << s.wall_ns << ",\"results_ns\":" << s.results_ns
                  << ",\"host_prepare_ns\":" << h.prepare_ns << ",\"h2d_submit_ns\":" << h.h2d_submit_ns
                  << ",\"h2d_wait_ns\":" << h.h2d_wait_ns << ",\"kernel_submit_ns\":" << h.kernel_submit_ns
                  << ",\"kernel_wait_ns\":" << h.kernel_wait_ns << ",\"d2h_submit_ns\":" << h.d2h_submit_ns
                  << ",\"d2h_wait_ns\":" << h.d2h_wait_ns << ",\"unpack_ns\":" << h.unpack_ns
                  << ",\"h2d_ms\":" << d.h2d_ms << ",\"kernel_ms\":" << d.kernel_ms
                  << ",\"d2h_ms\":" << d.d2h_ms << ",\"event_read_ns\":" << d.read_ns << "}\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const NearestEnvironment environment;
#if defined(__x86_64__)
        require((_mm_getcsr() & 0x8040U) == 0, "host FTZ/DAZ must be off");
#endif
        require(argc >= 2, "usage: vectorforge_cuda_prepared --gate|--host-gate|--budget|--direct-g Q|--prepared-g Q");
        const std::string_view mode = argv[1];
        if (argc == 2 && mode == "--budget") {
            check_budget();
            return 0;
        }
        const auto f = make_fixture(Layout::scattered);
        identity(f, "scattered");
        if (argc == 2 && mode == "--gate") {
            check_prepared(f);
        } else if (argc == 2 && mode == "--host-gate") {
            for (std::size_t j = 0; j < query_count; ++j) {
                exact(execute_scalar_reference(*f.snapshot, f.queries[j], predicates, top_k), f.expected.ranked[j]);
            }
            std::cout << "{\"kind\":\"host_gate\",\"pass\":true}\n";
        } else {
            require(argc == 3 && (mode == "--direct-g" || mode == "--prepared-g"), "explicit Contract G route required");
            const std::string_view horizon = argv[2];
            std::size_t count = 0;
            const auto parsed = std::from_chars(horizon.begin(), horizon.end(), count);
            require(parsed.ec == std::errc{} && parsed.ptr == horizon.end() &&
                    std::ranges::find(horizons, count) != horizons.end(), "unsupported query horizon");
            trial(f, count, mode == "--prepared-g");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "prepared_study_failed: " << error.what() << '\n';
        return 1;
    }
}
