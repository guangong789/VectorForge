#include "topk.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    namespace bench = vectorforge::benchmark;
    namespace topk = bench::topk;
    try {
        std::vector<std::string_view> arguments;
        for (int i = 1; i < argc; ++i) { arguments.emplace_back(argv[i]); }
        auto options = topk::parse_options(arguments);
        if (options.help) {
            std::cout << "Usage: vectorforge_topk_benchmark [--warmup W] [--iterations I>=3] [--seed S]\n"
                         "  Default: ten controlled Top-K cases, CSV on stdout.\n"
                         "  Any workload option selects one custom case:\n"
                         "    --candidates M --k K --distribution random|ties|ascending|descending\n"
                         "  --smoke: fixed tiny correctness/CSV run, not performance evidence.\n"
                         "  Performance requires an optimized unsanitized build without fast-math.\n";
            return 0;
        }
        const auto info = bench::metadata();
        bench::require_performance_build(info, options.smoke);
        auto workloads = topk::default_workloads();
        if (options.smoke) {
            std::cerr << "SMOKE ONLY: Top-K timings are not performance evidence.\n";
            options.warmup = 1;
            options.iterations = 3;
            workloads = {{"smoke", 32, 5, topk::Distribution::ties}};
        } else if (options.custom) { workloads = {*options.custom}; }
        topk::write_csv_header(std::cout);
        for (const auto& workload : workloads) {
            const auto run = topk::run_workload(workload, options);
            topk::write_csv(std::cout, workload, options, info, run);
        }
        if (!std::cout) { throw std::runtime_error("failed to write Top-K CSV"); }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "topk benchmark: " << error.what() << '\n';
        return 1;
    }
}
