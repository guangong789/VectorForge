#include "fma_comparison.hpp"

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
            std::cout << "Usage: vectorforge_fma_benchmark [--warmup W] [--iterations I>=3] [--seed S]\n"
                         "  Compare generic x86-64 and scalar AVX/FMA targets; alternating order.\n"
                         "  Default N=1024/8192, D=32/128/768; same data, CSV stdout.\n"
                         "  --rows N --dimension D selects a custom case (defaults N=1024, D=32).\n"
                         "  --smoke: tiny correctness case; no performance claims.\n"
                         "  Unsupported AVX/FMA: exit 77 without calling hardware code.\n";
            return 0;
        }
        if (!sim::fma::hardware_supported()) {
            std::cerr << "SKIP: AVX/FMA unavailable; hardware kernel was not called.\n";
            return 77;
        }
        const auto info = bench::metadata();
        bench::require_performance_build(info, options.sampling.smoke);
        auto workloads = sim::default_workloads();
        if (options.sampling.smoke) {
            std::cerr << "SMOKE ONLY: FMA comparison timings are not performance evidence.\n";
            options.sampling.warmup = 1;
            options.sampling.iterations = 3;
            workloads = {{"smoke", 17, 5}};
        } else if (options.custom) { workloads = {*options.custom}; }
        sim::fma::write_csv_header(std::cout);
        for (const auto& workload : workloads) {
            const auto run = sim::fma::run_workload(workload, options);
            sim::fma::write_csv(std::cout, workload, options, info, run);
        }
        if (!std::cout) { throw std::runtime_error("failed to write FMA comparison CSV"); }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FMA comparison: " << error.what() << '\n';
        return 1;
    }
}
