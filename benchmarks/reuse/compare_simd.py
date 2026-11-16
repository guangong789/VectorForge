"""Bounded E2E comparison using existing reuse benchmark executables; no dependencies.

python3 benchmarks/reuse/compare_simd.py --scalar /path/to/baseline/vectorforge_reuse_benchmark \
    --auto /path/to/trial/vectorforge_reuse_benchmark --output /tmp/simd-e2e-results
Historical comparison utility: it does not select a scoring kernel. Current
production query binaries always score generically, regardless of the legacy
trial marker. Replaying the old SIMD E2E comparison requires archived explicit
trial executables; live SIMD experiments use the dedicated kernel benchmarks.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scalar', required=True, type=Path)
    parser.add_argument('--auto', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case', nargs=3, type=int, action='append', metavar=('D', 'SELECTIVITY', 'BATCH'),
                        help='limit confirmation runs to explicitly chosen cases; repeat as needed')
    args = parser.parse_args()
    executables = {k: getattr(args, k).resolve(strict=True) for k in ('scalar', 'auto')}
    args.output.mkdir(parents=True, exist_ok=False)
    workloads = [(d, s, 256) for d in (32, 128, 768) for s in (1, 100)]
    workloads += [(128, 100, 7), (128, 1, 8192)]
    if args.case:
        workloads = [tuple(case) for case in args.case]
        if len(set(workloads)) != len(workloads) or any(d <= 0 or not 0 <= s <= 100 or b <= 0 for d, s, b in workloads):
            parser.error('cases must be distinct, D/batch positive, selectivity in 0..100')
    settings = dict(rows=8192, k=10, seed=1447440437, warmup=2, iterations=8, pairs=4,
                    workloads=workloads, order='rotate cases; alternate first executable exactly 2/2 per case',
                    binaries={k: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                              for k, p in executables.items()},
                    investigation_gate='paired median >10% AND >0.005ms, with repeated paired evidence; no statistical-confidence claim')
    (args.output / 'settings.json').write_text(json.dumps(settings, indent=2))
    records = {case: [] for case in workloads}
    for pair in range(settings['pairs']):
        cases = workloads[pair:] + workloads[:pair]
        for d, selectivity, batch in cases:
            results = {}
            order = ('scalar', 'auto') if pair % 2 == 0 else ('auto', 'scalar')
            common = ['--rows', '8192', '--dimension', str(d), '--selectivity', str(selectivity),
                      '--k', '10', '--batch-size', str(batch), '--reuse', '1', '--warmup', '2',
                      '--iterations', '8', '--seed', '1447440437']
            for variant in order:
                run = subprocess.run([str(executables[variant]), *common], check=True, capture_output=True, text=True)
                name = f'{pair}-d{d}-s{selectivity}-b{batch}-{variant}'
                (args.output / (name + '.csv')).write_text(run.stdout)
                if run.stderr:
                    (args.output / (name + '.stderr')).write_text(run.stderr)
                rows = list(csv.DictReader(io.StringIO(run.stdout)))
                if len(rows) != 1 or rows[0]['sanitized'] != '0' or rows[0]['build_type'] != 'Release':
                    raise RuntimeError('expected exactly one unsanitized Release workload')
                results[variant] = rows[0]
            for field in ('dataset_checksum', 'result_checksum', 'qualifying_candidates', 'returned_results',
                          'compiler', 'optimization_flags', 'cpu', 'os', 'seed'):
                if results['scalar'][field] != results['auto'][field]:
                    raise RuntimeError(f'incomparable runs: {field}')
            records[(d, selectivity, batch)].append(results)
    summary = []
    for case, pairs in records.items():
        result = dict(zip(('d', 'selectivity', 'batch'), case))
        for path in ('legacy', 'trusted'):
            field = path + '_query_median_ms'
            a = [float(p['scalar'][field]) for p in pairs]
            b = [float(p['auto'][field]) for p in pairs]
            ratios = [y / x for x, y in zip(a, b)]
            deltas = [y - x for x, y in zip(a, b)]
            result[path] = dict(scalar_median_ms=statistics.median(a), auto_median_ms=statistics.median(b),
                                paired_ratio_median=statistics.median(ratios), paired_ratio_range=[min(ratios), max(ratios)],
                                paired_ratios=ratios, paired_delta_median_ms=statistics.median(deltas),
                                scalar_run_medians=a, auto_run_medians=b,
                                material_slow_pairs=sum(r > 1.1 and d > 0.005 for r, d in zip(ratios, deltas)))
        summary.append(result)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Completed {len(workloads)} cases, four balanced pairs each; raw CSVs and summary in {args.output}')


if __name__ == '__main__':
    main()
