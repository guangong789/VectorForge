#include "accumulator.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    namespace bench = vectorforge::benchmark;
    namespace study = bench::accumulator;
    try {
        std::vector<std::string_view> arguments;
        for (int i = 1; i < argc; ++i) { arguments.emplace_back(argv[i]); }
        const bool ranking_only = arguments.size() == 1 && arguments[0] == "--ranking";
        auto options = ranking_only ? study::Options{} : study::parse_options(arguments);
        if (options.sampling.help) {
            std::cout << "Usage: vectorforge_accumulator_benchmark [--rows N] [--dimension D]\n"
                         "  [--warmup W] [--iterations even I>=4] [--seed S]\n"
                         "  Default: N=1024/8192, D=32/128/768; two warm-ups/eight measured scans.\n"
                         "  Sequential hardware FMA versus four-way hardware FMA; CSV stdout.\n"
                         "  --ranking (alone): four fixed close-score/cancellation/control cases; no timing.\n"
                         "  --smoke: tiny correctness/CSV case, not performance evidence.\n"
                         "  Unsupported AVX/FMA: exit 77 without calling hardware code.\n";
            return 0;
        }
        if (!bench::similarity::fma::hardware_supported()) {
            std::cerr << "SKIP: AVX/FMA unavailable; no hardware kernel called.\n";
            return 77;
        }
        const auto info = bench::metadata();
        if (ranking_only) {
            study::write_ranking_csv(std::cout, info, study::ranking_study());
        } else {
            bench::require_performance_build(info, options.sampling.smoke);
            auto workloads = bench::similarity::default_workloads();
            if (options.sampling.smoke) {
                std::cerr << "SMOKE ONLY: accumulator timings are not performance evidence.\n";
                options.sampling.warmup = 1;
                options.sampling.iterations = 4;
                workloads = {{"smoke", 17, 5}};
            } else if (options.custom) { workloads = {*options.custom}; }
            study::write_csv_header(std::cout);
            for (const auto& workload : workloads) {
                const auto run = study::run_workload(workload, options);
                study::write_csv(std::cout, workload, options, info, run);
            }
        }
        if (!std::cout) { throw std::runtime_error("failed to write accumulator CSV"); }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "accumulator study: " << error.what() << '\n';
        return 1;
    }
}
