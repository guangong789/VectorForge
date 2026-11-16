#pragma once

// Compile-time protection for core floating-point validation and sequential FMA.
// Used by the configure probe AND the actual scoring/admission/predicate TUs, so
// late target or inherited options cannot bypass it when these macros expose
// their effect. Do not silently override caller options with -fno-fast-math.
//
// Covers recognised compiler macros, not every compiler-specific relaxation or
// external application's link flags/runtime floating-point environment (rounding,
// FTZ/DAZ). In particular, options without diagnostic macros may remain undetected.
// GCC macro coverage is tested; Clang coverage depends on its exposed macros.
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0) || \
    defined(__ASSOCIATIVE_MATH__) || defined(__RECIPROCAL_MATH__) || \
    defined(__NO_SIGNED_ZEROS__) || defined(_M_FP_FAST)
#error "VectorForge numerical contract rejects fast/finite/unsafe math; preserve sequential float32 FMA and finite checks"
#endif
