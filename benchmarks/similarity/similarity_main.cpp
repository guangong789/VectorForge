#include "similarity.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    namespace bench = vectorforge::benchmark;
    namespace sim = bench::similarity;
    try {
        std::vector<std::string_view> arguments;
        for (int i = 1; i < argc; ++i) { arguments.emplace_back(argv[i]); }
        auto options = sim::parse_options(arguments);
        if (options.sampling.help) {
            std::cout << "Usage: vectorforge_similarity_benchmark [--warmup W] [--iterations I>=3] [--seed S]\n"
                         "  Default: N=1024/8192, D=32/128/768; sequential float32 FMA; CSV stdout.\n"
                         "  --rows N --dimension D: one custom case (defaults N=1024, D=32).\n"
                         "  --smoke: fixed tiny correctness case, not performance evidence.\n"
                         "  Performance requires an optimized unsanitized build without fast-math.\n";
            return 0;
        }
        const auto info = bench::metadata();
        bench::require_performance_build(info, options.sampling.smoke);
        auto workloads = sim::default_workloads();
        if (options.sampling.smoke) {
            std::cerr << "SMOKE ONLY: similarity timings are not performance evidence.\n";
            options.sampling.warmup = 1;
            options.sampling.iterations = 3;
            workloads = {{"smoke", 17, 5}};
        } else if (options.custom) { workloads = {*options.custom}; }
        sim::write_csv_header(std::cout);
        for (const auto& workload : workloads) {
            const auto run = sim::run_workload(workload, options);
            sim::write_csv(std::cout, workload, options, info, run);
        }
        if (!std::cout) { throw std::runtime_error("failed to write similarity CSV"); }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "similarity benchmark: " << error.what() << '\n';
        return 1;
    }
}
