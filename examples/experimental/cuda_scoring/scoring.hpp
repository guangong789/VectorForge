#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace cuda_scoring_pilot {

using RowIndex = std::uint32_t;
inline constexpr auto no_failure = std::numeric_limits<std::uint32_t>::max();

struct Score {
    float value;                    // +0 on failure, interpreted only when status succeeds.
    std::uint32_t first_failure;    // Zero-based dimension, or no_failure.
};

void select_device(bool verbose = true);

struct ResidencyStats {
    std::uint64_t allocation_ns = 0;
    std::uint64_t upload_ns = 0;
    std::size_t free_device_bytes = 0;
    std::size_t embedding_bytes = 0;
    std::size_t device_bytes = 0;
    std::size_t pinned_bytes = 0;
};

// Disjoint host-wall intervals. A wait includes elapsed device work and host
// synchronization overhead; it is not a CUDA Event measurement.
struct ScorePhases {
    std::uint64_t prepare_ns = 0;
    std::uint64_t h2d_submit_ns = 0;
    std::uint64_t h2d_wait_ns = 0;
    std::uint64_t kernel_submit_ns = 0;
    std::uint64_t kernel_wait_ns = 0;
    std::uint64_t d2h_submit_ns = 0;
    std::uint64_t d2h_wait_ns = 0;
    std::uint64_t unpack_ns = 0;
};

#ifdef VECTORFORGE_V2_LOCALITY
struct DevicePhases {
    float h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;
    std::uint64_t read_ns = 0;
};
#endif

// Contract G. Finite float32 operands, +0 start, round-to-nearest FMA in
// dimension order, gradual underflow and a finite check after every step.
// Each ordered slot is independent, including duplicates. Host FP flags,
// rounding controls, errno, math providers and CPU prefix/exception ordering
// are outside this contract. A numerical failure is a status, never a score.
// Owns one CUDA stream, device buffers and pinned staging. Use synchronously
// from one host thread on the device selected by select_device().
class ResidentScorer {
public:
    ResidentScorer(std::span<const float> row_major, std::size_t rows,
                   std::size_t dimension, std::size_t candidate_capacity,
                   bool verbose = true, ResidencyStats* residency = nullptr);
    ~ResidentScorer();

    [[nodiscard]] std::vector<Score> score(std::span<const RowIndex> candidates,
                                           std::span<const float> query, ScorePhases* phases = nullptr
#ifdef VECTORFORGE_V2_LOCALITY
                                           , DevicePhases* device = nullptr
#endif
                                           );
#ifdef VECTORFORGE_V2_LOCALITY
    // Allocate the diagnostic events outside the query interval on this owner.
    void enable_device_timing();
#endif
    void close();

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace cuda_scoring_pilot
