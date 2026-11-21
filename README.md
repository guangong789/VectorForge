简体中文 | [English](README.en.md)

# VectorForge

**面向精确过滤向量检索的 C++20 引擎**

VectorForge 在内存中按类型化列式元数据过滤记录，对按行连续存储的 FP32 向量计算精确点积 Top-K。项目以 C++ 数据所有权、数值正确性和性能工程为重点。生产库在 CPU 上执行，可选的独立 CUDA 实验研究 GPU 评分和数据驻留。

## 架构

~~~mermaid
flowchart LR
    input["查询与数据"]
    subgraph cpu_path["生产 CPU · Contract S"]
        cpu_filter["CPU 谓词过滤"] --> cpu_score["CPU 标量评分"] --> cpu_topk["CPU 有界堆 Top-K"]
    end
    subgraph cuda_path["CUDA 实验 · Contract G"]
        gpu_filter["CPU 谓词过滤"] --> gpu_score["GPU 评分（向量已驻留）"] --> gpu_topk["CPU 校验与 Top-K"]
    end
    input --> cpu_filter
    input --> gpu_filter
~~~

## C++ 核心设计

- **所有权与视图。** `ColumnBatch` 和 `EmbeddingBuffer` 通过 RAII 管理存储。`BatchView` 和只读 `std::span` 借用底层数据，切批时无需复制。查询结果独立持有 RecordID 和分数。
- **不可变快照。** `ValidatedDataset` 在准入时深拷贝数据，检查行数对齐、浮点值有限和 RecordID 唯一性。可信查询复用这些检查结果，每次仍验证查询向量和谓词。
- **有界排名。** 批式执行器逐批过滤，每次最多缓冲 32 个候选行号和对应分数。全局堆按需分配，最多保留 `min(K, M)` 项，`M` 为命中数。候选窗口与排名的额外存储为 `O(32 + min(K, M))`。数据集存储、准入分配和原始输入验证另计。独立标量参考收集所有命中结果再全排序，用于差分验证。两条路径都按分数降序、同分按 RecordID 升序输出。
- **流式聚合。** COUNT、SUM 和融合 Affine SUM 可直接处理数值列，成功路径无需动态分配。SUM 和 Affine SUM 以 `int32_t` 列为输入，按行顺序累加到 `int64_t`，逐步检查溢出。

## 数值语义与性能工程

- **CPU Contract S。** 点积从 `+0.0f` 开始，按维度顺序执行 FP32 FMA，每一步都检查中间值的有限性。编译要求禁用 fast-math。合同还保留可观察的验证与失败顺序，以及适用的浮点环境、数学函数行为、异常标志和 `errno`。差分测试逐位比较分数，并核对结果顺序。
- **CUDA Contract G。** 实验路径采用独立合同。对有限 FP32 输入，按维度顺序执行舍入到最近值的 FMA，保留渐进下溢，并记录每个候选首个非有限中间值所在的维度。CPU 校验返回状态后再发布 Top-K。 [CUDA 实验说明](examples/experimental/cuda_scoring/README.md)
- **可信查询对照。** 在一组 WSL2 Release 测量中，固定 `N=8192`、`D=128`、100% 命中、`K=10`、批大小 256，引入独立评分器和固定候选窗口后，可信查询耗时从 **1.93 ms 降至 1.00 ms**（六组配对运行的各次中位数再取中位数，计入完整查询，准入单独计时）。收益幅度取决于入口和负载。 [CPU 基准与复现方法](benchmarks/README.md#bounded-production-candidate-scoring-v1-06)

## 快速开始

在仓库根目录运行。CPU 构建只需 CMake 3.20 或更新版本及支持 C++20 的编译器。

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
~~~

## 设计决策与边界

- **布局贴合访问模式。** 列式元数据让过滤沿所需列扫描，按行存放向量则让单条记录的维度连续，便于逐维评分。
- **用一次准入换取验证复用。** 不可变快照隔离原始存储的别名。重复查询省去全量数据验证和 ID 去重，能否摊薄首次复制成本取决于单次查询节省的时间和复用次数。 [准入与复用基准](benchmarks/README.md#validated-admission-and-query-reuse-v0-08)
- **将排名内存与命中量解耦。** 小 K 时，有界堆无需物化全部命中结果。K 接近命中数时，空间优势缩小，维护堆也有成本。 [独立 Top-K 对比](benchmarks/README.md#isolated-top-k-comparison-v0-06)
- **生产评分固定走通用标量。** 跨行 SIMD 可以维持单条记录的 FMA 顺序，但已有反例在分数位模式和浮点异常标志相同时改变了 `errno`。SIMD 保留在独立实验中。 [SIMD 验证记录](src/scoring/simd_integration_notes.md)
- **执行范围有限。** CPU 核心面向单机内存中的精确扫描、排名和聚合。CUDA 独立实验研究驻留评分、数据传输和查询复用，自动放置原型限于已验证的历史负载与环境。

## 仓库布局

| 路径 | 内容 |
| --- | --- |
| [include/vectorforge/](include/vectorforge/)、[src/](src/) | C++ 公共 API 与 CPU 实现 |
| [tests/](tests/) | 数据、执行器和数值合同测试 |
| [benchmarks/](benchmarks/) | CPU 查询、排名与评分实验 |
| [examples/integration/ai_retrieval/](examples/integration/ai_retrieval/) | 独立 C++ 消费者与检索正确性验证 |
| [examples/experimental/cuda_scoring/](examples/experimental/cuda_scoring/) | 独立 CUDA 评分、驻留和放置实验 |
