#pragma once

#include "scoring.hpp"
#include "vectorforge/batched_executor.hpp"

#include <memory>

namespace placement_cuda {

struct Resident;
enum class State { ready, closed, invalid, device_context, capacity };

struct Preparation {
    cuda_scoring_pilot::ResidencyStats memory;
    std::int64_t context_ns = 0, pack_ns = 0, residency_ns = 0;
    int device = -1;
    std::uintptr_t context = 0;
};

struct CandidateEvidence {
    std::vector<cuda_scoring_pilot::RowIndex> rows;
    std::vector<cuda_scoring_pilot::Score> scores;
};

// This fixed private boundary keeps the CUDA fatbinary out of the CPU executable.
// The resident query takes no snapshot argument.
struct Api {
    Resident* (*create)(std::shared_ptr<const vectorforge::ValidatedDataset>, std::size_t, Preparation&);
    void (*destroy)(Resident*) noexcept;
    void (*close)(Resident&);
    State (*status)(const Resident&) noexcept;
    bool (*owns)(const Resident&, const std::shared_ptr<const vectorforge::ValidatedDataset>&) noexcept;
    std::vector<vectorforge::ScoredRecord> (*query)(
        Resident&, std::span<const float>, std::span<const vectorforge::Predicate>,
        std::size_t, std::size_t, CandidateEvidence*);
};

} // namespace placement_cuda

extern "C" const placement_cuda::Api* vectorforge_placement_cuda_api();
