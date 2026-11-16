#include <cerrno>
#include <cstddef>
#include <dlfcn.h>
#include <limits>

// Test-only ELF interposer. The real fmaf supplies score/IEEE behavior; this
// provider adds deterministic range-error reporting for underflow to zero.
namespace {
using Fma = float (*)(float, float, float);
Fma next_fma() {
    static const auto next = reinterpret_cast<Fma>(dlsym(RTLD_NEXT, "fmaf"));
    return next;
}
std::size_t calls = 0; // The diagnostic executable is single-threaded.
}
extern "C" bool vectorforge_test_fmaf_ready() noexcept { return next_fma() != nullptr; }
extern "C" std::size_t vectorforge_test_fmaf_calls() noexcept { return calls; }
extern "C" float fmaf(float x, float y, float z) noexcept {
    const auto next = next_fma();
    if (next == nullptr) { errno = ENOSYS; return std::numeric_limits<float>::quiet_NaN(); }
    ++calls;
    const float value = next(x, y, z);
    if (value == 0.0F && x != 0.0F && y != 0.0F && z == 0.0F) { errno = ERANGE; }
    return value;
}
