#include "benchmark.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    namespace bench = vectorforge::benchmark;
    try {
        std::vector<std::string_view> arguments;
        for (int i = 1; i < argc; ++i) {
            arguments.emplace_back(argv[i]);
        }
        auto options = bench::parse_options(arguments);
        if (options.help) {
            std::cout << "Usage: vectorforge_benchmark [--warmup W] [--iterations I>=3] [--seed S]\n"
                         "  Default: eight small workloads, CSV on stdout.\n"
                         "  Any workload option selects one custom case:\n"
                         "    --rows N --dimension D --selectivity PERCENT --k K --batch-size B\n"
                         "  --smoke: fixed tiny correctness/CSV run; not performance evidence.\n"
                         "  Performance runs require an optimized, unsanitized build without fast-math.\n";
            return 0;
        }
        const auto info = bench::metadata();
        bench::require_performance_build(info, options.smoke);
        auto workloads = bench::default_workloads();
        if (options.smoke) {
            std::cerr << "SMOKE ONLY: timings are not performance evidence.\n";
            options.warmup = 1;
            options.iterations = 3;
            workloads = {{"smoke", 32, 4, 25, 3, 7}};
        } else if (options.custom) {
            workloads = {*options.custom};
        }
        bench::write_csv_header(std::cout);
        for (const auto& workload : workloads) {
            const auto run = bench::run_workload(workload, options);
            bench::write_csv(std::cout, workload, options, info, run);
        }
        if (!std::cout) {
            throw std::runtime_error("failed to write benchmark CSV");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark: " << error.what() << '\n';
        return 1;
    }
}
