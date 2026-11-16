# Private cross-row SIMD research checkpoint (V1-07)

**Final decision B: scalar production policy, retained SIMD research.**
`score_candidates()` unconditionally delegates to `score_candidates_scalar()`.
Trusted and legacy public queries use the original generic implementation, with
its sequential FMA, completed-prefix, FP-state and actual math-provider behavior.
No option or trial marker can redirect that entry to SIMD. Public APIs, W=32,
executors, predicates, COUNT, Scalar Reference, admission and ranking are unchanged.
The isolated SIMD/state-replay implementation is retained for explicit private
tests and dedicated benchmarks only. Re-enablement needs a new reviewed contract
and release gate; existing speedups are not an authorization to reopen it.

## Experimental preconditions and provider boundary

The experiment was validated on Linux x86-64, GCC 13.4, glibc 2.35 headers and
runtime, ordinary glibc libm with the inspected hardware `__fmaf_fma3` behavior,
AVX2/FMA and OSXSAVE/XCR0 XMM+YMM support. Build/runtime version checks restrict
experimental execution; they do not authorize production dispatch. CPU/OS and platform capabilities are cached;
thread FP state is read on every explicit experimental scoring invocation. Nearest rounding, FTZ/DAZ
off and masked SSE/x87 exceptions are mandatory. Other controls/CPUs/platforms
use the original generic scorer within the experiment. Production uses generic
scalar regardless of these checks. Groups smaller than eight bypass probing.

Compiler/libc versions, linked libm and actual fmaf binding are different facts.
The checkpoint reproducer substituted fmaf, preserved score bits/IEEE flags and
set errno=ERANGE for multiplication underflow to zero. Eligibility still returned
true: original scalar ended with errno=34, SIMD retained initial errno=7. Failure
rollback cannot repair provider effects missing from a successful SIMD group.
This is the release blocker that withdrew V1-07D's initial A conclusion. Provider
identity/path/version alone would not prove behavior either. No provider-detection
framework is added; production remains scalar for every provider.
Other libc/toolchain behavior, asynchronous mutation of the same thread's FP
environment, and floating-point traps are not claimed compatible SIMD cases.
Unsupported FP controls fall back without being changed. An older/non-x86 CPU
was not physically available; native-excluded builds and final linked code were
checked instead. Finite tests do not constitute a universal proof.

The native object has private AVX2/FMA options, explicit `-fno-lto`, and no static
initializer. Generic dispatch is compiled with `-frounding-math -fno-lto`; no
specialized ISA options enter generic code. The legacy advanced
`VECTORFORGE_TEST_SIMD_DISPATCH` marker remains accepted for old reproduction
commands and test reporting, but is not passed to production code and cannot
change its scalar policy. `VECTORFORGE_BUILD_AVX2_SCORING` controls only compilation
of the private object, not production selection. It can exclude the native object
for generic-build validation; neither option is a public runtime selector.

## Address and ordered FP-state/error handling

EmbeddingBuffer construction proves row_count*D fits size_t and equals allocation
length. Before native entry, every selected row is bounded and checked without
multiplication first: row*D + D-1 <= PTRDIFF_MAX/sizeof(float). On this x86-64
configuration that also bounds signed 64-bit element indices scaled by four.
All gathered addresses belong to the same allocation as the row-zero pointer.
Two four-lane i64 gathers process exactly eight valid rows; no inactive lane
accesses memory. Sparse, repeated and nonconsecutive global rows retain slot
order. No Record ID participates in gather addressing.

Each group starts its lanes at +0 and applies float32 FMA in dimension order,
checking finiteness at every step. Generic dispatch checkpoints `fenv_t` and
errno before each group. A successful group retains its legitimate effects. A
failed group restores its own entry environment/errno, then replays with the
original generic scorer in candidate order. Pre-existing flags and earlier
successful groups therefore survive; speculative later-lane flags do not.
Only the affected eight-row group replays, never all N/M candidates. One local
checkpoint plus eight int64 offsets is bounded independently of dataset size;
existing W=32 arrays and the global heap are reused. Scalar tails never gather.

The inspected glibc 2.35 fegetenv/fesetenv implementations return zero for valid
objects and preserve the required controls while saving/restoring all six x87
and MXCSR status bits, including denormal operand. This is a specific verified
libc assumption, not a generic portable FP-state framework. No exit-time blanket
clear, global FP-state cache, signal handler or rounding/mask change is used.

Invalid rows or unrepresentable gather offsets choose scalar processing rather
than throwing during preflight. Existing shape diagnostics and candidate-order
failures remain authoritative. The unchanged executor consumes the completed
prefix before propagating its existing overflow diagnostic, retaining earlier
candidate-allocation failure precedence. No partial query result escapes.

## Historical blocker and integrated validation

V1-07B's finite counterexample uses query=[0.5,1,2], row0=[1,0,FLT_MAX],
row1=[denorm_min,0,0], with eight candidate rows. Scalar stops at row0 dimension2;
SIMD speculates row1 dimension0, leaving FE_UNDERFLOW and denormal-operand state.
Standard flags were 40 versus 56; raw MXCSR status was 40 versus 58. V1-07C proved
bounded checkpoint/replay locally. V1-07D integrates that method; the unisolated
negative control remains only in study code so the original defect still reproduces.

The historical Release core tests passed with the then-experimental query trial.
The expanded FP study
passed 911 checks (including expected historical-control mismatches), comparing
score bits/canaries, prefix, C++ diagnostics, FE flags, six raw status bits in each
unit, FP controls and errno. Success, legitimate underflow, incoming flags,
multiple failures, earlier successful groups and sequential/nested checkpoints
are covered. Public trusted/legacy queries also run across changing controls.
Allocation precedence and near-tie/Record-ID mapping tests remain exact. Focused
ASan/UBSan/LSan passed; gather safety relies on explicit bounds, not assuming ASan
instruments every hardware gather lane. Generic/native instruction isolation was
checked in final linked binaries.

## Production provider-substitution regression

Unsanitized native Linux/ELF test builds add `scalar_provider_normal` and
`scalar_provider_substitution`. A small test-only fmaf interposer counts calls,
delegates numeric computation to real fmaf, and reports ERANGE for nonzero finite
multiplication underflowing to zero. The test verifies actual interception, then
compares original generic scoring against production, including score bits,
completed prefixes, C++ error diagnostics, standard/raw FP state and errno.
Trusted/legacy query controls compile the existing executor source with only
entry/scorer symbols renamed to fix generic scoring, preserving the exact same
validation, heap/allocation and failure order. K=0, no matches, malformed inputs,
partial failure and incoming flags are covered; the legacy marker cannot disable
production checks. No new public API or math-provider dependency enters the engine.

Whole-static, cross and sanitizer configurations omit this ELF diagnostic.
Runtime absence of an ELF dynamic loader, or a compiler-expanded/direct-bound fmaf
that cannot be intercepted, is explicitly skipped rather than claimed covered.
Other generic builds remain valid. Existing low-level SIMD/FP implementations are
unchanged by closure, allowing their matching targeted sanitizer evidence to be
reused. README files remain untouched.

## Historical E2E evidence and explicit experimental access

Release builds with `VECTORFORGE_BUILD_BENCHMARKS=ON` and
`VECTORFORGE_BUILD_CROSS_ROW_STUDY=ON` retain `vectorforge_cross_row_benchmark`
and `vectorforge_fp_state_study --benchmark`, which call private kernels directly.
CPU/FP guards remain mandatory. These are the live experimental entry points;
production query benchmarks always use generic scoring now.

`benchmarks/reuse/compare_simd.py` preserves the historical comparison method and
binary manifests. Replaying the old SIMD E2E evidence needs the archived explicit
trial executables; the legacy marker cannot recreate an SIMD query binary now.
No new experimental query hook is introduced during closure. The eight-case matrix covers D=32/128/768, 1%/100% selectivity, full and
partial windows, trusted/legacy paths and sparse full windows. It uses four
balanced pairs, warmup=2, iterations=8, seed=1447440437, with exact reference
verification outside timing. Metadata, fingerprints, binary hashes, raw samples,
relative/absolute changes and variability remain in temporary build outputs.

In the historical unpinned i9-12900H/GCC 13.4 WSL2 experiment (not current
production execution), high-selectivity trusted paired
state-safe auto/scalar median ratios were 0.664/0.654/0.523 for D=32/128/768;
legacy ratios were 0.777/0.764/0.656. Capture costs are included. Initial batch=7
trusted ratio was 1.085 (+83us); a focused comparison against the current forced
scalar build gave 0.986 (range 0.851-1.013), so that regression did not reproduce
consistently. Low-selectivity D=768 trusted likewise changed from 1.065 to 0.987
in focused confirmation. No material repeatable >10% AND >0.005ms paired
regression was demonstrated. No universal gain or causal attribution to SIMD
alone follows: this path also replaces generic fmaf calls with hardware FMA.
These results, and earlier V1-07B measurements without isolation, remain research
evidence only. They do not describe current production speedups or establish
provider-independent observable equivalence.
