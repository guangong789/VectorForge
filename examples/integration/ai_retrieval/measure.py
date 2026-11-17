#!/usr/bin/env python3
"""Fixed V1-12C campaign: correctness gate, 32 timing workers, 8 separate memory workers.

Run with Python -B. Only standard modules are used. All evidence goes to a new
directory outside the checkout. Allocation-call counts are deliberately unavailable:
the existing test probes replace global new/delete and are not linked into this binary.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import time
import traceback

CELLS = {
    "N10K-S1": (10000, 100, 99174, 12594456802194367061),
    "N10K-S90": (10000, 9000, 9020946, 17980169636896482800),
    "N100K-S1": (100000, 1000, 992331, 13302898193871766348),
    "N100K-S90": (100000, 90000, 91898078, 9034647294887284646),
}
QUERY_FP = 1173672798050634943
PATHS = ("bounded", "full-sort")
BUDGET = 512 * 1024 * 1024
SECONDS = 180


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fp(words):
    value = 14695981039346656037
    for word in words:
        for byte in int(word).to_bytes(8, "little"):
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def quantile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def rss(pid):
    try:
        text = Path(f"/proc/{pid}/status").read_text()
        return {key: int(re.search(rf"^{key}:\s+(\d+) kB", text, re.M).group(1)) * 1024
                for key in ("VmRSS", "VmHWM")}
    except (OSError, AttributeError):
        return None


class Campaign:
    def __init__(self, binary, output, root):
        self.binary, self.output, self.root = binary.resolve(), output.resolve(), root.resolve()
        check(not self.output.is_relative_to(self.root), "evidence must be outside Git")
        self.output.mkdir(parents=True, exist_ok=False)
        self.ids, self.completed = set(), []
        self.binary_sha = digest(self.binary)
        names = self.git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
        self.sources = {name: digest(self.root / name) for name in names}
        self.status = self.git("status", "--porcelain=v1", "--untracked-files=all")
        self.golden = {}
        self.summaries, self.query_rows, self.process_rows, self.memory_rows = [], [], [], []
        self.save("source-hashes.json", {"revision": self.git("rev-parse", "HEAD").strip(), "files": self.sources})
        self.source_manifest_sha = digest(self.output / "source-hashes.json")
        shutil.copy2(self.binary, self.output / "workload-binary")
        shutil.copy2(__file__, self.output / "measure.py")

    def git(self, *args):
        return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=self.root, text=True)

    def save(self, name, data):
        (self.output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    def unchanged(self):
        check(digest(self.binary) == self.binary_sha, "binary changed after correctness gate")
        check(self.git("status", "--porcelain=v1", "--untracked-files=all") == self.status, "working tree changed")
        check(all(digest(self.root / n) == h for n, h in self.sources.items()), "source changed")

    def execute(self, run_id, args, mode, deadline=None):
        check(run_id not in self.ids, "duplicate run ID")
        self.ids.add(run_id)
        self.unchanged()
        command = [str(self.binary), *args]
        record = {"run_id": run_id, "argv": command, "mode": mode, "binary_sha256": self.binary_sha,
                  "source_manifest_sha256": self.source_manifest_sha, "validated": False}
        self.save(run_id + ".meta.json", record)
        start = time.monotonic()
        cadence = .005 if mode == "memory" else .020
        samples, missing, observed, combined_max, child_max, os_hwm = 0, 0, 0, 0, 0, 0
        sampling = (self.output / (run_id + ".rss.jsonl")).open("w") if mode == "memory" else None
        failure = None
        with (self.output / (run_id + ".stdout")).open("w") as stdout, (self.output / (run_id + ".stderr")).open("w") as stderr:
            process = subprocess.Popen(command, cwd=self.root, stdout=stdout, stderr=stderr)
            try:
                while process.poll() is None:
                    stamp = time.monotonic_ns()
                    child, parent = rss(process.pid), rss(os.getpid())
                    samples += 1
                    if child is None or parent is None:
                        missing += 1
                    else:
                        observed += 1
                        combined = child["VmRSS"] + parent["VmRSS"]
                        combined_max = max(combined_max, combined)
                        child_max = max(child_max, child["VmRSS"])
                        os_hwm = max(os_hwm, child["VmHWM"])
                        if sampling:
                            sampling.write(json.dumps({"timestamp_ns": stamp, "child_rss": child["VmRSS"],
                                                       "parent_rss": parent["VmRSS"], "combined": combined}) + "\n")
                        check(combined <= BUDGET, "combined child/supervisor resident memory exceeded 512 MiB")
                    check(time.monotonic() - start < 120, "worker timeout")
                    check(deadline is None or time.monotonic() < deadline, "180-second campaign budget exceeded")
                    time.sleep(cadence)
            except BaseException as error:
                failure = str(error)
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            finally:
                process.wait()
                if sampling:
                    sampling.close()
        record.update(exit_code=process.returncode, elapsed_seconds=time.monotonic() - start,
                      observed_combined_peak_bytes=combined_max, sampled_child_peak_bytes=child_max,
                      sampled_os_hwm_bytes=os_hwm, sample_count=samples, missing_samples=missing,
                      sampling_cadence_seconds=cadence, failure=failure)
        self.save(run_id + ".meta.json", record)
        check(failure is None, failure)
        check(process.returncode == 0, run_id + " unexpected exit; raw stdout/stderr retained")
        check(not (self.output / (run_id + ".stderr")).read_text().strip(), run_id + " unexpected stderr")
        check(observed > 0, run_id + " no resident-memory budget observations")
        check(deadline is None or time.monotonic() <= deadline, "180-second campaign budget exceeded")
        self.unchanged()
        return record, (self.output / (run_id + ".stdout")).read_text()

    def read_expectations(self, cell, path):
        fields = path.read_text().split()
        n, m, tokens, data_fp = CELLS[cell]
        check(fields[:2] == ["VF_EXPECTATIONS_1", cell], "invalid expectations format")
        check(tuple(map(int, fields[2:7])) == (n, m, tokens, data_fp, QUERY_FP), "accepted dataset/query/aggregate changed")
        cursor, results = 7, []
        for q in range(8):
            index, expected = map(int, fields[cursor:cursor + 2]); cursor += 2
            check(index == q, "invalid expected query index")
            words = [20]
            ids = []
            for _ in range(20):
                identity, bits = map(int, fields[cursor:cursor + 2]); cursor += 2
                check(0 <= bits < 2**32, "invalid expected score bits")
                words.extend((identity, bits)); ids.append(identity)
            check(len(set(ids)) == 20 and fp(words) == expected, "corrupt independent expected output")
            results.append(expected)
        check(cursor == len(fields), "trailing expectations data")
        return {"path": str(path), "sha256": digest(path), "query_results": results,
                "session_fp": fp([m, tokens, *results])}

    def gate(self):
        for cell in CELLS:
            path = self.output / (cell + ".expected")
            record, text = self.execute("gate-" + cell, [cell, "--export", str(path)], "correctness")
            check([int(q) for q in re.findall(r"query=(\d+) exact_top_k=20 three_way_agreement=true", text)] == list(range(8)), "incomplete three-way gate")
            check(text.count("PASS cell=" + cell) == 1, "missing correctness completion")
            self.golden[cell] = self.read_expectations(cell, path)
            record["validated"] = True
            self.save("gate-" + cell + ".meta.json", record)
        self.save("correctness-binding.json", {"binary_sha256": self.binary_sha, "source_manifest_sha256": self.source_manifest_sha, "cells": self.golden})

    def validate_records(self, run_id, cell, path, mode, text):
        rows = [json.loads(line) for line in text.splitlines()]
        golden = self.golden[cell]
        check(digest(golden["path"]) == golden["sha256"], "expectations changed")
        n, m, tokens, data_fp = CELLS[cell]
        for row in rows:
            check((row["run_id"], row["cell"], row["path"], row["mode"]) == (run_id, cell, path, mode), "inconsistent record identity")
            check((row["n"], row["m"], row["count"], row["token_sum"], row["data_fp"], row["query_fp"], row["d"], row["k"], row["batch"]) ==
                  (n, m, m, tokens, data_fp, QUERY_FP, 128, 20, 1024), "record workload drift")
            expected = golden["query_results"][row["query"]] if row["record"] == "query" else golden["session_fp"]
            check(row["result_fp"] == expected, "result fingerprint changed")
        if mode == "memory":
            stages = ("before_construction", "after_construction", "after_admission", "after_source_release", "after_sessions", "after_snapshot_release")
            check(len(rows) == 7 and [r["record"] for r in rows] == ["checkpoint"] * 6 + ["complete"], "incomplete memory run")
            check([r["stage"] for r in rows[:6]] == list(stages), "invalid memory stage order")
            check(all(r["rss_bytes"] > 0 and r["os_hwm_bytes"] > 0 for r in rows[:6]), "missing memory counters")
            check(all(a["timestamp_ns"] <= b["timestamp_ns"] for a, b in zip(rows[:5], rows[1:6])), "memory timestamps reversed")
            check(rows[-1]["sessions"] == 10 and rows[-1]["queries"] == 80 and rows[-1]["allocation_calls"] is None, "invalid memory completion")
            check(len(rows[-1]["result_capacities_bytes"]) == 8, "missing result capacities")
            return rows
        cursor = 0
        def take(kind, **fields):
            nonlocal cursor
            check(cursor < len(rows), "truncated timing output")
            row = rows[cursor]; cursor += 1
            check(row["record"] == kind and all(row.get(k) == v for k, v in fields.items()), "invalid timing record order")
            if "duration_ns" in row:
                check(row["duration_ns"] == row["end_ns"] - row["start_ns"] and row["duration_ns"] > 0 and row["start_ns"] >= 0, "invalid interval")
            return row
        a = take("phase", stage="construction"); b = take("phase", stage="admission"); c = take("phase", stage="source_release")
        check(a["end_ns"] <= b["start_ns"] <= b["end_ns"] <= c["start_ns"], "lifecycle order changed")
        previous = c["end_ns"]
        for period in ("initial", "steady"):
            if period == "steady":
                take("warmup", sessions=2)
            for i in range(10):
                session = take("session", period=period, session=i)
                check(session["start_ns"] >= previous, "sessions overlap or reverse")
                prior = session["start_ns"]
                for q in range(8):
                    query = take("query", period=period, session=i, query=q)
                    check(prior <= query["start_ns"] <= query["end_ns"] <= session["end_ns"], "query outside session")
                    check(query["result_capacity_bytes"] >= 20 * query["scored_record_bytes"], "invalid result capacity")
                    prior = query["end_ns"]
                previous = session["end_ns"]
                if period == "initial" and i in (0, 9):
                    prefix = take("prefix", stage="one_session" if i == 0 else "ten_sessions")
                    check(prefix["start_ns"] == b["start_ns"] and prefix["end_ns"] == session["end_ns"], "prefix reconstructed or incorrect")
        end = take("phase", stage="snapshot_release")
        check(previous <= end["start_ns"], "snapshot released before query completion")
        take("complete", sessions=22, steady_sessions=10, queries=176)
        check(cursor == len(rows), "extra timing records")
        return rows

    def worker(self, cell, path, run_id, mode, deadline):
        args = [cell, "--mode", mode, "--path", path, "--run-id", run_id, "--expectations", self.golden[cell]["path"]]
        record, text = self.execute(run_id, args, mode, deadline)
        try:
            rows = self.validate_records(run_id, cell, path, mode, text)
        except BaseException as error:
            record["failure"] = "evidence validation: " + str(error)
            self.save(run_id + ".meta.json", record)
            raise
        record["validated"] = True
        self.save(run_id + ".meta.json", record)
        self.completed.append(run_id)
        if mode == "memory":
            self.memory_rows.append({"cell": cell, "path": path, "run_id": run_id, "checkpoints": rows[:6],
                                     "os_hwm_bytes": max(record["sampled_os_hwm_bytes"], *(r["os_hwm_bytes"] for r in rows[:6])),
                                     "sampled_max_rss_bytes": record["sampled_child_peak_bytes"], "observed_combined_peak_bytes": record["observed_combined_peak_bytes"],
                                     "sample_count": record["sample_count"], "missing_samples": record["missing_samples"],
                                     "result_capacities_bytes": rows[-1]["result_capacities_bytes"], "allocation_calls": None})
        else:
            sessions = [r["duration_ns"] for r in rows if r["record"] == "session" and r["period"] == "steady"]
            process = {"cell": cell, "path": path, "run_id": run_id, "session_p50_ns": quantile(sessions, .5), "session_p95_ns": quantile(sessions, .95),
                       "steady_total_ns": sum(sessions), "sequential_sessions_per_second": 10e9 / sum(sessions)}
            process.update({r["stage"] + "_ns": r["duration_ns"] for r in rows if r["record"] in ("phase", "prefix")})
            self.process_rows.append(process)
            for row in rows:
                if row["record"] in ("session", "query") and row["period"] == "steady":
                    self.query_rows.append(row)
        print(run_id + " verified", flush=True)

    def summarize(self):
        for cell in CELLS:
            for path in PATHS:
                values = [r for r in self.query_rows if r["cell"] == cell and r["path"] == path]
                sessions = [r["duration_ns"] for r in values if r["record"] == "session"]
                check(len(sessions) == 40, "incomplete session population")
                processes = [r for r in self.process_rows if r["cell"] == cell and r["path"] == path]
                check(len(processes) == 4, "incomplete process population")
                result = {"cell": cell, "path": path, "samples": 40, "session_p50_ns": quantile(sessions, .5), "session_p95_ns": quantile(sessions, .95),
                          "sequential_sessions_per_second": 40e9 / sum(sessions),
                          "per_process_session_p50_ns": [p["session_p50_ns"] for p in processes]}
                for phase in ("construction", "admission", "source_release", "snapshot_release", "one_session", "ten_sessions"):
                    result[phase + "_median_ns"] = statistics.median(p[phase + "_ns"] for p in processes)
                result["ten_session_amortized_median_ns"] = statistics.median(p["ten_sessions_ns"] / 10 for p in processes)
                result["queries"] = []
                for q in range(8):
                    times = [r["duration_ns"] for r in values if r["record"] == "query" and r["query"] == q]
                    check(len(times) == 40, "incomplete query population")
                    result["queries"].append({"query": q, "samples": 40, "p50_ns": quantile(times, .5), "p95_ns": quantile(times, .95)})
                self.summaries.append(result)
        self.save("timing-summary.json", self.summaries)
        self.save("process-summary.json", self.process_rows)
        self.save("memory-summary.json", self.memory_rows)

    def run(self):
        check(platform.system() == "Linux", "resident-memory measurement requires Linux /proc")
        compile_file = self.binary.parent / "compile_commands.json"
        cache = (self.binary.parent / "CMakeCache.txt").read_text()
        commands = json.loads(compile_file.read_text())
        own = [r for r in commands if r["file"].endswith("/ai_retrieval/workload.cpp")]
        check(len(own) == 1 and "-O3" in own[0]["command"] and "-DNDEBUG" in own[0]["command"], "optimized Release binary required")
        check(not any(x in own[0]["command"] for x in ("-ffast-math", "-funsafe-math", "-ffinite-math", "-fsanitize", "-march=native")), "invalid measurement compiler options")
        check("VECTORFORGE_BUILD_AVX2_SCORING:BOOL=OFF" in cache and "VECTORFORGE_ENABLE_SANITIZERS:BOOL=OFF" in cache, "SIMD/sanitizers must be disabled")
        shutil.copy2(compile_file, self.output / "compile_commands.json")
        shutil.copy2(self.binary.parent / "CMakeCache.txt", self.output / "CMakeCache.txt")
        topology = {p.parent.name: {q: (p / q).read_text().strip() for q in ("physical_package_id", "core_id", "thread_siblings_list")}
                    for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology")}
        self.save("metadata.json", {"binary_sha256": self.binary_sha, "source_manifest_sha256": self.source_manifest_sha,
                  "compiler_commands": own, "cpuinfo": Path("/proc/cpuinfo").read_text(), "topology": topology,
                  "affinity": sorted(os.sched_getaffinity(0)), "platform": platform.platform(), "python": platform.python_version(),
                  "monotonic_clock": vars(time.get_clock_info("monotonic")), "memory_budget_bytes": BUDGET,
                  "notes": "un-pinned sequential WSL/Linux processes; timing budget guard polls RSS at20ms, separate memory samples at5ms; parent overhead included in simultaneous observed sums; no sums of independent peaks; allocation calls unavailable; capacity is not peak workspace",
                  "sampling": "4 processes x10 steady sessions; 10 initial +2 warmups before steady; linear (n-1)*p quantiles; preserve outliers; nested clocks included in sessions"})
        self.gate()
        self.start = time.monotonic()
        deadline = self.start + SECONDS
        order = []
        names = list(CELLS)
        for repeat in range(4):
            for offset in range(4):
                cell = names[(repeat + offset) % 4]
                paths = PATHS if repeat % 2 == 0 else tuple(reversed(PATHS))
                for path in paths:
                    run_id = f"t{repeat}_{cell}_{path}"
                    order.append({"run_id": run_id, "cell": cell, "path": path, "mode": "timing"})
        for cell in CELLS:
            for path in PATHS:
                order.append({"run_id": f"m0_{cell}_{path}", "cell": cell, "path": path, "mode": "memory"})
        self.save("run-order.json", order)
        for item in order:
            self.worker(item["cell"], item["path"], item["run_id"], item["mode"], deadline)
        self.summarize()
        self.save("results.json", {"status": "passed", "timing_processes": 32, "memory_processes": 8, "correctness_gates": 4,
                  "completed_run_ids": self.completed, "campaign_elapsed_seconds": time.monotonic() - self.start,
                  "runtime_budget_seconds": SECONDS, "all_source_binary_expected_fingerprints_bound": True})
        self.save("evidence-file-hashes.json", {str(p.relative_to(self.output)): digest(p) for p in sorted(self.output.rglob("*")) if p.is_file() and p.name != "evidence-file-hashes.json"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    campaign = Campaign(args.binary, args.output, args.source_root)
    try:
        campaign.run()
    except BaseException as error:
        campaign.save("results.json", {"status": "failed", "error": str(error), "completed_run_ids": campaign.completed})
        campaign.save("evidence-file-hashes.json", {str(p.relative_to(campaign.output)): digest(p) for p in sorted(campaign.output.rglob("*")) if p.is_file() and p.name != "evidence-file-hashes.json"})
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
