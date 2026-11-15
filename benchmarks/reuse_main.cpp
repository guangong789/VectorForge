#include "reuse.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    namespace bench = vectorforge::benchmark;
    namespace reuse = bench::reuse;
    try {
        std::vector<std::string_view> arguments;
        for (int i = 1; i < argc; ++i) { arguments.emplace_back(argv[i]); }
        auto options = reuse::parse_options(arguments);
        if (options.sampling.help) {
            std::cout << "Usage: vectorforge_reuse_benchmark [--warmup W] [--iterations I>=3] [--seed S]\n"
                         "  Default: four datasets, modeled R=1/10/100 workflow totals, CSV on stdout.\n"
                         "  Any workload/reuse option selects one custom case:\n"
                         "    --rows N --dimension D --selectivity PERCENT --k K --batch-size B --reuse R>0\n"
                         "  --smoke: fixed tiny correctness/CSV case, not performance evidence.\n"
                         "  Measures admission and legacy/trusted queries; totals/break-even are models.\n";
            return 0;
        }
        const auto info = bench::metadata();
        bench::require_performance_build(info, options.sampling.smoke);
        auto workloads = reuse::default_workloads();
        if (options.sampling.smoke) {
            std::cerr << "SMOKE ONLY: reuse timings are not performance evidence.\n";
            options.sampling.warmup = 1;
            options.sampling.iterations = 3;
            workloads = {{{"smoke", 32, 4, 25, 3, 7}, {2}}};
        } else if (options.sampling.custom || options.custom_reuse) {
            workloads = {{options.sampling.custom.value_or(bench::Workload{"custom"}),
                          {options.custom_reuse.value_or(10)}}};
        }
        reuse::write_csv_header(std::cout);
        for (const auto& workload : workloads) {
            const auto run = reuse::run_workload(workload, options);
            reuse::write_csv(std::cout, workload, options, info, run);
        }
        if (!std::cout) { throw std::runtime_error("failed to write reuse CSV"); }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "reuse benchmark: " << error.what() << '\n';
        return 1;
    }
}
