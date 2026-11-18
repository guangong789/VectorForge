#include "scoring.hpp"

#include <cudaHelper/cuda_check.cuh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cuda_scoring_pilot {
namespace {

static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4);
static_assert(sizeof(Score) == 8 && std::is_trivially_copyable_v<Score>);
constexpr std::size_t device_limit = 128 * 1024 * 1024;
constexpr std::size_t pinned_limit = 4 * 1024 * 1024;
constexpr std::size_t free_memory_floor = 512 * 1024 * 1024;
constexpr std::size_t upload_chunk = 1024 * 1024;
constexpr unsigned threads_per_block = 128;
enum Buffer : std::size_t { embeddings, indices, query, scores, buffer_count };
constexpr std::array buffer_names{"embeddings", "indices", "query", "scores"};

std::uint64_t time_ns(bool enabled) {
    if (!enabled) { return 0; }
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

void check(cudaError_t status, const char* phase) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cuda_failure phase=%s code=%d name=%s\n",
                     phase, static_cast<int>(status), cudaGetErrorName(status));
        cuda_helper::check_cuda(status, phase, __FILE__, __LINE__);
    }
}

std::size_t multiply(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::length_error("scoring byte size overflows size_t");
    }
    return a * b;
}

bool finite(std::span<const float> input) {
    return std::all_of(input.begin(), input.end(), [](float value) { return std::isfinite(value); });
}

__global__ void score_rows(const float* row_major, std::size_t dimension, const RowIndex* candidate_rows,
                           const float* query_values, std::size_t count, Score* output) {
    const auto slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= count) { return; }
    const auto* row = row_major + static_cast<std::size_t>(candidate_rows[slot]) * dimension;
    float score = +0.0f;
    for (std::uint32_t d = 0; d < dimension; ++d) {
        score = __fmaf_rn(query_values[d], row[d], score);
        if (!isfinite(score)) {
            output[slot] = {0.0f, d};
            return;
        }
    }
    output[slot] = {score, no_failure};
}

bool kernel_error(cudaError_t status) {
    return status == cudaErrorLaunchFailure || status == cudaErrorIllegalAddress ||
           status == cudaErrorLaunchTimeout || status == cudaErrorAssert ||
           status == cudaErrorIllegalInstruction || status == cudaErrorMisalignedAddress;
}

} // namespace

struct ResidentScorer::State {
    std::size_t rows = 0;
    std::size_t dimension = 0;
    std::size_t capacity = 0;
    std::array<void*, buffer_count> dev{};
    std::array<std::size_t, buffer_count> bytes{};
    void* staging = nullptr;
    std::size_t staging_bytes = 0;
    cudaStream_t stream = nullptr;
    bool usable = false;
    bool verbose = true;

    ~State() noexcept {
        try { close(); }
        catch (const std::exception& error) {
            std::fprintf(stderr, "cleanup unconfirmed: %s\n", error.what());
            std::fflush(nullptr);
            std::_Exit(74);
        }
    }

    void sync(const char* phase) const {
        const auto status = cudaStreamSynchronize(stream);
        check(status, kernel_error(status) ? "asynchronous kernel failure" : phase);
    }

    void upload(Buffer buffer, const void* input, std::size_t count) const {
        check(cudaMemcpyAsync(dev[buffer], input, count, cudaMemcpyHostToDevice, stream),
              "H2D submit");
        if (verbose) { std::printf("transfer H2D buffer=%s bytes=%zu\n", buffer_names[buffer], count); }
    }

    void close() {
        if (stream == nullptr) { return; }
        usable = false;
        const auto drained = cudaStreamSynchronize(stream);
        if (drained != cudaSuccess) {
            // No host/device buffer is freed while its stream may still reference
            // it. This standalone process cannot safely continue with the context.
            std::fprintf(stderr, "cleanup stream completion unconfirmed code=%d name=%s\n",
                         static_cast<int>(drained), cudaGetErrorName(drained));
            std::fflush(nullptr);
            std::_Exit(74);
        }
        auto first_error = cudaSuccess;
        const auto released = [&](cudaError_t status, const char* name, std::size_t size) {
            if (verbose || status != cudaSuccess) {
                std::printf("release buffer=%s bytes=%zu status=%s\n", name, size, cudaGetErrorName(status));
            }
            if (first_error == cudaSuccess) { first_error = status; }
        };
        for (std::size_t i = dev.size(); i > 0; --i) {
            const auto index = i - 1;
            if (dev[index]) {
                released(cudaFree(std::exchange(dev[index], nullptr)), buffer_names[index], bytes[index]);
            }
        }
        if (staging) {
            released(cudaFreeHost(std::exchange(staging, nullptr)), "pinned-staging", staging_bytes);
        }
        released(cudaStreamDestroy(std::exchange(stream, nullptr)), "stream", 0);
        check(first_error, "resource cleanup");
    }
};

void select_device(bool verbose) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "device discovery");
    if (count == 0) { throw std::runtime_error("no visible CUDA device"); }
    check(cudaSetDevice(0), "select device 0");
    cudaDeviceProp property{};
    check(cudaGetDeviceProperties(&property, 0), "device properties");
    int driver = 0, runtime = 0;
    check(cudaDriverGetVersion(&driver), "driver version");
    check(cudaRuntimeGetVersion(&runtime), "runtime version");
    if (verbose) {
        std::printf("device=%s visible_count=%d sm=%d.%d driver_api=%d runtime_api=%d total_bytes=%zu\n",
                    property.name, count, property.major, property.minor, driver, runtime, property.totalGlobalMem);
    }
    if (property.major != 8 || property.minor != 6) {
        throw std::runtime_error("this pilot was compiled for sm_86");
    }
}

ResidentScorer::ResidentScorer(std::span<const float> row_major, std::size_t rows,
                               std::size_t dimension, std::size_t candidate_capacity,
                               bool verbose, ResidencyStats* residency)
    : state_(std::make_unique<State>()) {
    auto& s = *state_;
    s.verbose = verbose;
    if (dimension == 0 || dimension > no_failure || rows > no_failure || candidate_capacity > no_failure) {
        throw std::invalid_argument("unsupported scoring dimensions or row-index range");
    }
    const auto elements = multiply(rows, dimension);
    s.bytes = {multiply(elements, sizeof(float)), multiply(candidate_capacity, sizeof(RowIndex)),
               multiply(dimension, sizeof(float)), multiply(candidate_capacity, sizeof(Score))};
    std::size_t requested = 0;
    for (const auto size : s.bytes) {
        if (size > device_limit - requested) { throw std::length_error("device allocation budget exceeded"); }
        requested += size;
    }
    s.staging_bytes = std::max(std::min(s.bytes[embeddings], upload_chunk), requested - s.bytes[embeddings]);
    if (s.staging_bytes > pinned_limit) { throw std::length_error("pinned staging budget exceeded"); }
    if (row_major.size() != elements || !finite(row_major)) {
        throw std::invalid_argument("row-major embeddings must match the shape and be finite");
    }

    s.rows = rows;
    s.dimension = dimension;
    s.capacity = candidate_capacity;
    const auto allocation_start = time_ns(residency != nullptr);
    std::size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total), "device memory query before allocations");
    if (verbose) {
        std::printf("memory free_bytes=%zu total_bytes=%zu device_request_bytes=%zu pinned_request_bytes=%zu\n",
                    free, total, requested, s.staging_bytes);
    }
    if (free < free_memory_floor || free < requested) {
        throw std::runtime_error("insufficient free GPU memory for pilot");
    }

    check(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking), "create private stream");
    for (std::size_t i = 0; i < s.dev.size(); ++i) {
        if (s.bytes[i] == 0) { continue; }
        check(cudaMalloc(&s.dev[i], s.bytes[i]), buffer_names[i]);
        if (verbose) {
            std::printf("allocation buffer=%s bytes=%zu status=cudaSuccess\n", buffer_names[i], s.bytes[i]);
        }
    }
    check(cudaMallocHost(&s.staging, s.staging_bytes), "allocate pinned staging");
    if (verbose) {
        std::printf("allocation buffer=pinned-staging bytes=%zu status=cudaSuccess\n", s.staging_bytes);
    }

    const auto upload_start = time_ns(residency != nullptr);
    const auto input = std::as_bytes(row_major);
    auto* destination = static_cast<std::byte*>(s.dev[embeddings]);
    for (std::size_t offset = 0; offset < input.size();) {
        const auto count = std::min(upload_chunk, input.size() - offset);
        std::memcpy(s.staging, input.data() + offset, count);
        check(cudaMemcpyAsync(destination + offset, s.staging, count, cudaMemcpyHostToDevice, s.stream),
              "resident H2D submit");
        if (verbose) { std::printf("transfer H2D buffer=embeddings bytes=%zu\n", count); }
        s.sync("resident H2D synchronization");
        offset += count;
    }
    if (residency) {
        *residency = {upload_start - allocation_start, time_ns(true) - upload_start,
                      free, s.bytes[embeddings], requested, s.staging_bytes};
    }
    s.usable = true;
}

ResidentScorer::~ResidentScorer() = default;

std::vector<Score> ResidentScorer::score(std::span<const RowIndex> candidates,
                                        std::span<const float> query_values, ScorePhases* phases) {
    auto phase_start = time_ns(phases != nullptr);
    const auto finish = [&](std::uint64_t ScorePhases::*field) {
        if (phases) {
            const auto end = time_ns(true);
            phases->*field = end - phase_start;
            phase_start = end;
        }
    };
    if (phases) { *phases = {}; }
    auto& s = *state_;
    if (!s.usable) { throw std::logic_error("scorer is closed or a CUDA operation failed"); }
    if (query_values.size() != s.dimension || !finite(query_values)) {
        throw std::invalid_argument("query must match the embedding dimension and be finite");
    }
    if (candidates.size() > s.capacity) { throw std::invalid_argument("candidate capacity exceeded"); }
    for (const auto row : candidates) {
        if (row >= s.rows) { throw std::out_of_range("candidate row is out of range"); }
    }
    if (candidates.empty()) { finish(&ScorePhases::prepare_ns); return {}; }

    std::vector<Score> result(candidates.size());
    auto* index_stage = static_cast<std::byte*>(s.staging);
    auto* query_stage = index_stage + s.bytes[indices];
    auto* output_stage = query_stage + s.bytes[query];
    const auto output_bytes = multiply(result.size(), sizeof(Score));
    std::memcpy(index_stage, candidates.data(), candidates.size_bytes());
    std::memcpy(query_stage, query_values.data(), query_values.size_bytes());
    s.usable = false; // A failed submission can leave queued work referencing staging.
    finish(&ScorePhases::prepare_ns);
    s.upload(indices, index_stage, candidates.size_bytes());
    s.upload(query, query_stage, query_values.size_bytes());
    finish(&ScorePhases::h2d_submit_ns);
    s.sync("query H2D synchronization");
    finish(&ScorePhases::h2d_wait_ns);

    const auto blocks = (candidates.size() - 1) / threads_per_block + 1;
    score_rows<<<static_cast<unsigned>(blocks), threads_per_block, 0, s.stream>>>(
        static_cast<const float*>(s.dev[embeddings]), s.dimension, static_cast<const RowIndex*>(s.dev[indices]),
        static_cast<const float*>(s.dev[query]), candidates.size(), static_cast<Score*>(s.dev[scores]));
    check(cudaGetLastError(), "kernel submit");
    finish(&ScorePhases::kernel_submit_ns);
    s.sync("kernel synchronization");
    finish(&ScorePhases::kernel_wait_ns);
    if (s.verbose) {
        std::printf("kernel status=completed blocks=%zu threads=%u candidates=%zu\n",
                    blocks, threads_per_block, candidates.size());
    }

    check(cudaMemcpyAsync(output_stage, s.dev[scores], output_bytes, cudaMemcpyDeviceToHost, s.stream), "D2H submit");
    if (s.verbose) { std::printf("transfer D2H buffer=scores bytes=%zu\n", output_bytes); }
    finish(&ScorePhases::d2h_submit_ns);
    s.sync("scores D2H synchronization");
    finish(&ScorePhases::d2h_wait_ns);
    std::memcpy(result.data(), output_stage, output_bytes);
    s.usable = true;
    finish(&ScorePhases::unpack_ns);
    return result;
}

void ResidentScorer::close() { state_->close(); }

} // namespace cuda_scoring_pilot
