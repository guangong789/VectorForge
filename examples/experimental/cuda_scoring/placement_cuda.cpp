#include "placement_cuda.hpp"

#include <thread>

#define VECTORFORGE_PRIVATE_RETRIEVAL_REUSE
#include "retrieval.cpp"

namespace placement_cuda {

namespace {
using GetContext = decltype(&cuCtxGetCurrent);
using GetDevice = decltype(&cuCtxGetDevice);

template<class Entry>
Entry driver_entry(const char* name) {
    void* entry = nullptr;
    cudaDriverEntryPointQueryResult status{};
    lifecycle_check_cuda(cudaGetDriverEntryPoint(name, &entry, cudaEnableDefault, &status), name);
    require(status == cudaDriverEntryPointSuccess && entry, "required current-context query unavailable");
    return reinterpret_cast<Entry>(entry);
}
} // namespace

struct Resident {
    Resident(std::shared_ptr<const ValidatedDataset> owned, std::size_t limit, Preparation& prep)
        : snapshot(std::move(owned)), capacity(limit) {
        const auto runtime = prepare_runtime(true);
        prep.context_ns = runtime.initialization + runtime.prelude;
        context = runtime.context;
        get_context = driver_entry<GetContext>("cuCtxGetCurrent");
        get_device = driver_entry<GetDevice>("cuCtxGetDevice");
        lifecycle_check_cuda(cudaGetDevice(&device), "read approved CUDA device ordinal");
        int count = 0, driver = 0, runtime_version = 0;
        cudaDeviceProp props{};
        lifecycle_check_cuda(cudaGetDeviceCount(&count), "count evidence devices");
        lifecycle_check_cuda(cudaGetDeviceProperties(&props, device), "read evidence device properties");
        lifecycle_check_cuda(cudaDriverGetVersion(&driver), "read evidence driver API");
        lifecycle_check_cuda(cudaRuntimeGetVersion(&runtime_version), "read evidence runtime API");
        if (count != 1 || device != 0 || std::string_view(props.name) != "NVIDIA GeForce RTX 3060 Laptop GPU" ||
            props.major != 8 || props.minor != 6 || props.totalGlobalMem != 6'441'926'656ULL ||
            driver != 13050 || runtime_version != 12040 || !context_matches()) {
            state = State::device_context;
            return;
        }
        prep.device = device;
        prep.context = reinterpret_cast<std::uintptr_t>(context);
        const auto start = now_ns();
        Tick packed = 0;
        {
            const auto flat = pack(*snapshot);
            packed = now_ns();
            const auto& embeddings = snapshot->embeddings();
            scorer.emplace(flat, embeddings.row_count(), embeddings.dimension(), capacity, false, &prep.memory);
        }
        prep.pack_ns = packed - start;
        prep.residency_ns = now_ns() - packed;
        // ResidentScorer checks byte products and budgets, and synchronizes the
        // initial upload before construction returns.
        state = State::ready;
    }

    Resident(const Resident&) = delete;
    Resident& operator=(const Resident&) = delete;
    Resident(Resident&&) = delete;
    Resident& operator=(Resident&&) = delete;

    ~Resident() noexcept {
        if (scorer && !context_matches()) { unsafe_cleanup(); }
        // ResidentScorer drains its private stream before releasing resources.
    }

    bool context_matches() const noexcept {
        CUcontext current = nullptr;
        CUdevice current_device = -1;
        return std::this_thread::get_id() == thread && get_context(&current) == CUDA_SUCCESS && current == context &&
               get_device(&current_device) == CUDA_SUCCESS && current_device == device;
    }

    [[noreturn]] static void unsafe_cleanup() noexcept {
        std::fputs("placement cleanup context unconfirmed\n", stderr);
        std::fflush(nullptr);
        std::_Exit(74);
    }

    const std::shared_ptr<const ValidatedDataset> snapshot;
    const std::size_t capacity;
    const std::thread::id thread = std::this_thread::get_id();
    std::optional<gpu::ResidentScorer> scorer;
    GetContext get_context = nullptr;
    GetDevice get_device = nullptr;
    CUcontext context = nullptr;
    CUdevice device = -1;
    State state = State::invalid;
};

namespace {
Resident* create(std::shared_ptr<const ValidatedDataset> snapshot, std::size_t capacity, Preparation& prep) {
    return new Resident(std::move(snapshot), capacity, prep);
}
void destroy(Resident* resident) noexcept { delete resident; }
void close(Resident& resident) {
    if (resident.scorer && !resident.context_matches()) { Resident::unsafe_cleanup(); }
    resident.state = State::closed;
    if (resident.scorer) {
        resident.scorer->close();
        resident.scorer.reset();
    }
}
State status(const Resident& resident) noexcept {
    if (resident.state != State::ready) { return resident.state; }
    if (!resident.context_matches()) { return State::device_context; }
    if (resident.capacity != resident.snapshot->columns().row_count()) { return State::capacity; }
    return State::ready;
}
bool owns(const Resident& resident, const std::shared_ptr<const ValidatedDataset>& snapshot) noexcept {
    const auto& bound = resident.snapshot;
    return bound.get() == snapshot.get() && !bound.owner_before(snapshot) && !snapshot.owner_before(bound);
}
std::vector<ScoredRecord> query(Resident& resident, std::span<const float> q, std::span<const Predicate> filters,
                                std::size_t k, std::size_t batch, CandidateEvidence* evidence) {
    require(status(resident) == State::ready, "resident session no longer usable");
    try {
        CandidateTrace trace;
        auto results = retrieve_gpu(*resident.snapshot, *resident.scorer, q, filters, k, batch, nullptr,
                                    evidence ? &trace : nullptr);
        if (evidence) { *evidence = {std::move(trace.rows), std::move(trace.scores)}; }
        return results;
    } catch (...) {
        resident.state = State::invalid;
        throw;
    }
}
} // namespace
} // namespace placement_cuda

extern "C" __attribute__((visibility("default"))) const placement_cuda::Api* vectorforge_placement_cuda_api() {
    using namespace placement_cuda;
    static constexpr Api api{create, destroy, close, status, owns, query};
    return &api;
}
