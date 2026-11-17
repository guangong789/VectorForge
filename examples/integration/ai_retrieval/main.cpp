#include <vectorforge/batched_executor.hpp>
#include <vectorforge/column_batch.hpp>
#include <vectorforge/embedding_buffer.hpp>
#include <vectorforge/filtered_count.hpp>
#include <vectorforge/filtered_sum.hpp>
#include <vectorforge/query_types.hpp>
#include <vectorforge/validated_dataset.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace vectorforge;
constexpr std::size_t tenant_column = 0, category_column = 1, updated_column = 2, tokens_column = 3;
constexpr std::size_t dimension = 4, record_count = 9;
constexpr std::array<std::size_t, 3> requested_k{0, 1, 20}, batch_sizes{1, 4, 20};

struct DocumentChunk {
    RecordID id;
    std::int32_t tenant_id, category;
    std::int64_t updated_at;
    std::int32_t token_count;
    std::array<float, dimension> embedding;
};

std::array<DocumentChunk, record_count> documents() {
    // Numeric tenant IDs are metadata, not authentication or authorization.
    // Rows 0 and 4 have identical vectors and reversed ID order across B=4 batches.
    return {{{41, 7, 2, 1'700'000'100, 120, {1, 0.5F, 0, -0.5F}},
             {7, 7, 3, 1'700'000'200, 80, {0.5F, 1, 0, 0}},
             {83, 9, 2, 1'700'000'300, 200, {0, 0.5F, 1, 0}},
             {12, 7, 2, 1'700'000'400, 60, {0.5F, 0, 0, 0}},
             {5, 7, 2, 1'700'000'500, 140, {1, 0.5F, 0, -0.5F}},
             {90, 9, 3, 1'700'000'600, 50, {-1, 0, 0.5F, 1}},
             {26, 9, 2, 1'700'000'700, 160, {0.5F, -0.5F, 0, 1}},
             {3, 7, 2, 1'700'000'800, 40, {0, 0, 0, 0}},
             {62, 7, 3, 1'700'000'900, 100, {1, 0, 0.5F, 0}}}};
}

struct Request {
    std::array<float, dimension> query;
    std::span<const Predicate> predicates; // Borrows caller-owned arrays in run().
};
struct Expected {
    std::vector<ScoredRecord> ranked;
    std::size_t count = 0;
    std::int64_t tokens = 0;
};

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

// Consumer-local oracle: plain application records, no library execution/reference helpers.
NumericConstant field(const DocumentChunk& row, std::size_t column) {
    switch (column) {
    case tenant_column: return row.tenant_id;
    case category_column: return row.category;
    case updated_column: return row.updated_at;
    case tokens_column: return row.token_count;
    default: throw std::out_of_range("oracle metadata column");
    }
}

template <typename T> bool compare(T value, T constant, Comparison comparison) {
    switch (comparison) {
    case Comparison::less: return value < constant;
    case Comparison::less_equal: return value <= constant;
    case Comparison::greater: return value > constant;
    case Comparison::greater_equal: return value >= constant;
    case Comparison::equal: return value == constant;
    case Comparison::not_equal: return value != constant;
    }
    throw std::invalid_argument("oracle comparison");
}

bool matches(const DocumentChunk& row, std::span<const Predicate> predicates) {
    for (const auto& predicate : predicates) {
        const auto value = field(row, predicate.column_index);
        const bool match = std::visit([&]<typename T>(T constant) {
            const auto* typed = std::get_if<T>(&value);
            if (!typed) { throw std::invalid_argument("oracle predicate type"); }
            return compare(*typed, constant, predicate.comparison);
        }, predicate.constant);
        if (!match) { return false; }
    }
    return true;
}

Expected reference(const std::array<DocumentChunk, record_count>& rows, const Request& request, std::size_t k) {
    for (const auto value : request.query) { require(std::isfinite(value), "oracle query must be finite"); }
    // Validate all predicates before row filtering, independently of matches/AND short circuits.
    for (const auto& predicate : request.predicates) {
        const auto value = field(rows.front(), predicate.column_index);
        std::visit([&]<typename T>(T constant) {
            if (!std::get_if<T>(&value)) { throw std::invalid_argument("oracle predicate type"); }
            if constexpr (std::same_as<T, float>) {
                if (!std::isfinite(constant)) { throw std::invalid_argument("oracle predicate constant"); }
            }
            (void)compare(T{}, constant, predicate.comparison);
        }, predicate.constant);
    }
    Expected result;
    for (const auto& row : rows) {
        if (!matches(row, request.predicates)) { continue; }
        ++result.count;
        const auto delta = static_cast<std::int64_t>(row.token_count);
        if ((delta > 0 && result.tokens > std::numeric_limits<std::int64_t>::max() - delta) ||
            (delta < 0 && result.tokens < std::numeric_limits<std::int64_t>::min() - delta)) {
            throw std::overflow_error("oracle token prefix overflow");
        }
        result.tokens += delta;
        if (k == 0) { continue; }
        float score = 0.0F;
        for (std::size_t feature = 0; feature < dimension; ++feature) {
            score = std::fma(request.query[feature], row.embedding[feature], score);
            if (!std::isfinite(score)) { throw std::overflow_error("oracle score overflow"); }
        }
        result.ranked.push_back({row.id, score});
    }
    std::sort(result.ranked.begin(), result.ranked.end(), [](const auto& left, const auto& right) {
        if (left.score != right.score) { return left.score > right.score; }
        return left.id < right.id;
    });
    if (result.ranked.size() > k) { result.ranked.resize(k); }
    return result;
}

void exact(std::span<const ScoredRecord> actual, std::span<const ScoredRecord> expected) {
    require(actual.size() == expected.size(), "retrieval result size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(actual[i].id == expected[i].id, "retrieval ID/order mismatch");
        require(std::bit_cast<std::uint32_t>(actual[i].score) == std::bit_cast<std::uint32_t>(expected[i].score),
                "retrieval score bits mismatch");
    }
}

std::size_t rejections = 0;
template <typename Error, typename Function> void rejects(Function function) {
    try { function(); }
    catch (const Error&) { ++rejections; return; }
    throw std::runtime_error("expected public API exception was not thrown");
}

class NearestRounding {
public:
    NearestRounding() : previous_(std::fegetround()) {
        require(previous_ != -1 && std::fesetround(FE_TONEAREST) == 0, "FE_TONEAREST is unavailable");
    }
    ~NearestRounding() { (void)std::fesetround(previous_); }
    NearestRounding(const NearestRounding&) = delete;
    NearestRounding& operator=(const NearestRounding&) = delete;
private:
    int previous_;
};

void run() {
    const NearestRounding rounding;
    // Ordinary finite, binary-exact fixture; no subnormal/FTZ-sensitive arithmetic.
    // Exact score bits here do not claim arbitrary libm-provider errno/FP-side-effect equivalence.
    const std::array tenant_category{Predicate{tenant_column, Comparison::equal, std::int32_t{7}},
                                    Predicate{category_column, Comparison::equal, std::int32_t{2}}};
    const std::array recent_other_tenant{Predicate{tenant_column, Comparison::equal, std::int32_t{9}},
                                       Predicate{updated_column, Comparison::greater_equal, std::int64_t{1'700'000'400}}};
    const std::array absent{Predicate{tenant_column, Comparison::equal, std::int32_t{999}}};
    const std::array<Request, 5> requests{{{{1, 0, 0, 0}, tenant_category},
                                         {{0, 1, 0, 1}, tenant_category},
                                         {{0, 0, 1, 1}, recent_other_tenant},
                                         {{1, 0, 0, 0}, absent},
                                         {{0.5F, -0.5F, 1, 0}, {}}}};
    std::array<std::array<Expected, requested_k.size()>, requests.size()> expected;
    std::vector<ScoredRecord> owned_result;
    std::size_t retrieval_checks = 0, aggregate_checks = 0;
    {
        std::optional<ValidatedDataset> snapshot;
        {
            const auto records = documents();
            for (std::size_t q = 0; q < requests.size(); ++q) {
                for (std::size_t k = 0; k < requested_k.size(); ++k) {
                    expected[q][k] = reference(records, requests[q], requested_k[k]);
                }
            }
            const std::array<ScoredRecord, 4> hand_ranked{{{5, 1}, {41, 1}, {12, 0.5F}, {3, 0}}};
            exact(expected[0][2].ranked, hand_ranked);
            require(expected[0][2].count == 4 && expected[0][2].tokens == 360, "candidate metadata hand values");
            require(expected[2][2].count == 2 && expected[2][2].tokens == 210, "timestamp metadata hand values");

            std::vector<std::int32_t> tenants, categories, tokens;
            std::vector<std::int64_t> timestamps;
            std::vector<float> values;
            std::vector<RecordID> ids;
            for (const auto& row : records) {
                tenants.push_back(row.tenant_id); categories.push_back(row.category);
                timestamps.push_back(row.updated_at); tokens.push_back(row.token_count);
                values.insert(values.end(), row.embedding.begin(), row.embedding.end()); ids.push_back(row.id);
            }
            std::vector<ColumnBatch::Column> columns;
            columns.emplace_back(std::move(tenants)); columns.emplace_back(std::move(categories));
            columns.emplace_back(std::move(timestamps)); columns.emplace_back(std::move(tokens));
            const ColumnBatch metadata(std::move(columns));
            const EmbeddingBuffer embeddings(record_count, dimension, std::move(values));
            snapshot.emplace(metadata, embeddings, ids);
            require(snapshot->columns().row_count() == record_count && snapshot->embeddings().row_count() == record_count,
                    "snapshot row alignment");
            require(snapshot->columns().column<std::int32_t>(tenant_column).data() != metadata.column<std::int32_t>(tenant_column).data()
                    && snapshot->embeddings().row(0).data() != embeddings.row(0).data()
                    && snapshot->record_ids().data() != ids.data(), "snapshot must copy its inputs");
            for (std::size_t row = 0; row < record_count; ++row) {
                require(snapshot->record_ids()[row] == records[row].id, "snapshot ID alignment");
            }

            auto duplicate_ids = ids; duplicate_ids.back() = duplicate_ids.front();
            rejects<std::invalid_argument>([&] { const ValidatedDataset bad(metadata, embeddings, duplicate_ids); });
            const EmbeddingBuffer fewer_rows(record_count - 1, dimension, std::vector<float>((record_count - 1) * dimension));
            rejects<std::invalid_argument>([&] { const ValidatedDataset bad(metadata, fewer_rows, ids); });
            rejects<std::invalid_argument>([&] { const EmbeddingBuffer bad(record_count, dimension, std::vector<float>(35)); });
            rejects<std::invalid_argument>([] { const EmbeddingBuffer bad(record_count, 0, {}); });
            std::vector<float> nonfinite(record_count * dimension, 0);
            nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
            const EmbeddingBuffer bad_values(record_count, dimension, std::move(nonfinite));
            rejects<std::invalid_argument>([&] { const ValidatedDataset bad(metadata, bad_values, ids); });
            rejects<std::invalid_argument>([] {
                const ColumnBatch bad({std::vector<std::int32_t>{1, 2}, std::vector<std::int64_t>{1}});
            });
        } // Original records, column/embedding owners, vectors and ID owners are all destroyed.

        for (std::size_t q = 0; q < requests.size(); ++q) {
            const auto& request = requests[q];
            for (const auto batch : batch_sizes) {
                for (std::size_t repeat = 0; repeat < 2; ++repeat) {
                    for (std::size_t k = 0; k < requested_k.size(); ++k) {
                        auto actual = execute_batched(*snapshot, request.query, request.predicates, requested_k[k], batch);
                        exact(actual, expected[q][k].ranked); ++retrieval_checks;
                        if (q == 0 && batch == 4 && repeat == 0 && requested_k[k] == 20) { owned_result = std::move(actual); }
                    }
                    require(count_where(*snapshot, request.predicates, batch) == expected[q][2].count, "candidate COUNT mismatch");
                    // Existing raw-column SUM over public snapshot columns, not a trusted SUM API.
                    // Corpus candidate token total is not the number of tokens passed to an LLM.
                    require(sum_where(snapshot->columns(), tokens_column, request.predicates, batch) == expected[q][2].tokens,
                            "candidate token SUM mismatch");
                    aggregate_checks += 2;
                }
            }
        }
        const auto& query = requests[0].query;
        const std::array wrong_type{Predicate{tenant_column, Comparison::equal, std::int64_t{7}}};
        const std::array bad_index{Predicate{4, Comparison::equal, std::int32_t{7}}};
        const std::array bad_comparison{Predicate{tenant_column, static_cast<Comparison>(99), std::int32_t{7}}};
        rejects<std::invalid_argument>([&] { (void)execute_batched(*snapshot, query, wrong_type, 0, 4); });
        rejects<std::invalid_argument>([&] { (void)execute_batched(*snapshot, query, tenant_category, 0, 0); });
        rejects<std::out_of_range>([&] { (void)execute_batched(*snapshot, query, bad_index, 0, 4); });
        rejects<std::invalid_argument>([&] { (void)execute_batched(*snapshot, query, bad_comparison, 0, 4); });
        rejects<std::invalid_argument>([&] { (void)execute_batched(*snapshot, std::array<float, 3>{1, 0, 0}, tenant_category, 0, 4); });
        auto nonfinite_query = query; nonfinite_query[0] = std::numeric_limits<float>::infinity();
        rejects<std::invalid_argument>([&] { (void)execute_batched(*snapshot, nonfinite_query, tenant_category, 0, 4); });
        rejects<std::invalid_argument>([&] { (void)count_where(*snapshot, wrong_type, 4); });
        rejects<std::out_of_range>([&] { (void)sum_where(snapshot->columns(), 4, tenant_category, 4); });
        rejects<std::invalid_argument>([&] { (void)sum_where(snapshot->columns(), updated_column, tenant_category, 4); });
        const std::array overflow_query{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 0.0F, 0.0F};
        rejects<std::overflow_error>([&] { (void)execute_batched(*snapshot, overflow_query, tenant_category, 1, 4); });
        snapshot.reset(); // No borrowed views were saved; only owning expected/results survive.
        require(!snapshot.has_value(), "snapshot destruction boundary");
    }
    exact(owned_result, expected[0][2].ranked);
    require(owned_result.size() == 4 && owned_result.front().id == 5, "owned result after snapshot destruction");
    std::cout << "AI retrieval consumer: 9 records, " << retrieval_checks << " exact retrieval checks, "
              << aggregate_checks << " COUNT/SUM checks, " << rejections
              << " expected rejections; original inputs released before queries; 4 owned results survive snapshot destruction\n";
}
} // namespace

int main() {
    try { run(); return 0; }
    catch (const std::exception& error) {
        std::cerr << "AI retrieval consumer failed: " << error.what() << '\n'; return 1;
    }
}
