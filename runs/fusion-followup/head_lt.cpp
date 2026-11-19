#include <cublasLt.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

// The replay is compiled against the exported ABI of its pinned Torch wheel.
namespace at::cuda {
cublasLtHandle_t getCurrentCUDABlasLtHandle();
size_t getChosenWorkspaceSize();
size_t getCUDABlasLtWorkspaceSize();
void* getCUDABlasLtWorkspace();
}

namespace {
constexpr auto bf16 = CUDA_R_16BF;
constexpr auto compute = CUBLAS_COMPUTE_32F;
constexpr auto scale = CUDA_R_32F;
using Config = std::array<int32_t, 9>;

void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(cublasLtGetStatusString(status));
    }
}

template <typename T>
T read(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
    T value{};
    check(cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &value, sizeof(value), nullptr));
    return value;
}

template <typename T>
bool set(cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr, T value) {
    return cublasLtMatmulAlgoConfigSetAttribute(&algo, attr, &value, sizeof(value)) == CUBLAS_STATUS_SUCCESS;
}

std::vector<uint32_t> capabilities(const cublasLtMatmulAlgo_t& algo,
                                   cublasLtMatmulAlgoCapAttributes_t attr) {
    size_t bytes = 0;
    check(cublasLtMatmulAlgoCapGetAttribute(&algo, attr, nullptr, 0, &bytes));
    if (bytes == 0) {
        return {0};
    }
    std::vector<uint32_t> values(bytes / sizeof(uint32_t));
    check(cublasLtMatmulAlgoCapGetAttribute(&algo, attr, values.data(), bytes, nullptr));
    return values;
}

uint32_t capability(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoCapAttributes_t attr) {
    uint32_t value = 0;
    check(cublasLtMatmulAlgoCapGetAttribute(&algo, attr, &value, sizeof(value), nullptr));
    return value;
}

Config get_config(const cublasLtMatmulAlgo_t& algo) {
    Config config{};
    for (size_t i = 0; i < config.size(); ++i) {
        auto attr = static_cast<cublasLtMatmulAlgoConfigAttributes_t>(i);
        config[i] = i < 7 ? read<int32_t>(algo, attr) : read<uint16_t>(algo, attr);
    }
    return config;
}

struct Candidate {
    cublasLtMatmulAlgo_t algo;
    Config config;
    int heuristic_rank;
    size_t workspace_size;
    float waves;
};

struct Plan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t matmul = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    std::set<Config> seen;
    std::vector<Candidate> candidates;
    int validation_attempts = 0;
    int supported_ids = 0;
    size_t workspace_limit = 0;

    ~Plan() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (matmul) cublasLtMatmulDescDestroy(matmul);
    }

    void add(const cublasLtMatmulAlgo_t& algo, int heuristic_rank) {
        Config config = get_config(algo);
        if (!seen.insert(config).second) return;
        // Keep accumulation and any separate reduction in the compute type.
        if (config[3] != CUBLASLT_REDUCTION_SCHEME_NONE &&
            config[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) return;
        ++validation_attempts;
        cublasLtMatmulHeuristicResult_t result{};
        auto status = cublasLtMatmulAlgoCheck(handle, matmul, a, b, c, c, &algo, &result);
        if (status == CUBLAS_STATUS_SUCCESS && result.state == CUBLAS_STATUS_SUCCESS && result.workspaceSize <= workspace_limit) {
            candidates.push_back({algo, config, heuristic_rank, result.workspaceSize, result.wavesCount});
        }
    }

    void heuristic(size_t workspace_size, int heuristic_limit) {
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &workspace_size, sizeof(workspace_size)));
        std::vector<cublasLtMatmulHeuristicResult_t> results(heuristic_limit);
        int count = 0;
        check(cublasLtMatmulAlgoGetHeuristic(handle, matmul, a, b, c, c, preference, results.size(), results.data(), &count));
        for (int i = 0; i < count; ++i) {
            if (results[i].state == CUBLAS_STATUS_SUCCESS) add(results[i].algo, i);
        }
    }

    void enumerate(uint32_t alignment, int heuristic_limit, int64_t k) {
        check(cublasLtMatmulPreferenceCreate(&preference));
        uint32_t mask = CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                   &mask, sizeof(mask)));
        for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
            check(cublasLtMatmulPreferenceSetAttribute(preference, attr, &alignment, sizeof(alignment)));
        }
        heuristic(0, heuristic_limit);
        if (workspace_limit) {
            const auto standard_lt_budget = at::cuda::getCUDABlasLtWorkspaceSize();
            if (standard_lt_budget > 0 && standard_lt_budget < workspace_limit) {
                heuristic(standard_lt_budget, heuristic_limit);
            }
            heuristic(workspace_limit, heuristic_limit);
        }

        std::array<int, 4096> ids{};
        check(cublasLtMatmulAlgoGetIds(handle, compute, scale, bf16, bf16, bf16, bf16,
                                       ids.size(), ids.data(), &supported_ids));
        if (supported_ids == static_cast<int>(ids.size())) {
            throw std::runtime_error("Algorithm ID list reached the search capacity");
        }
        for (int i = 0; i < supported_ids; ++i) {
            cublasLtMatmulAlgo_t base{};
            if (cublasLtMatmulAlgoInit(handle, compute, scale, bf16, bf16, bf16, bf16,
                                       ids[i], &base) != CUBLAS_STATUS_SUCCESS) continue;
            add(base, -1);
            const auto tiles = capabilities(base, CUBLASLT_ALGO_CAP_TILE_IDS);
            const auto stages = capabilities(base, CUBLASLT_ALGO_CAP_STAGES_IDS);
            const auto swizzle = capability(base, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
            const auto custom = capability(base, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX);
            const auto reductions = capability(base, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK);
            const auto split_k_supported = capability(base, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT);
            for (auto tile : tiles) {
                for (auto stage : stages) {
                    for (uint32_t swizzle_id = 0; swizzle_id <= std::min(swizzle, 1u); ++swizzle_id) {
                        for (uint32_t option = 0; option <= std::min(custom, 2u); ++option) {
                            auto algo = base;
                            if (!set(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, tile) ||
                                !set(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, stage) ||
                                !set(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swizzle_id) ||
                                !set(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, option)) continue;
                            add(algo, -1);
                            if (!workspace_limit || !split_k_supported ||
                                !(reductions & CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) continue;
                            const auto fp32_reduction = static_cast<uint32_t>(CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE);
                            if (!set(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, fp32_reduction)) continue;
                            for (int32_t splits : {2, 4, 8, 16}) {
                                if (k % splits == 0 && set(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, splits)) add(algo, -1);
                            }
                        }
                    }
                }
            }
        }
    }
};
}  // namespace

extern "C" size_t head_lt_version() { return cublasLtGetVersion(); }
extern "C" size_t head_lt_inherited_workspace_size() { return at::cuda::getChosenWorkspaceSize(); }
extern "C" void* head_lt_inherited_workspace() {
    try {
        return at::cuda::getCUDABlasLtWorkspace();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ATen workspace lookup failed: %s\n", error.what());
        return nullptr;
    }
}

extern "C" Plan* head_lt_create(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
                                int trans_b, uint32_t alignment, int heuristic_limit, size_t workspace_limit) {
    try {
        auto plan = std::make_unique<Plan>();
        plan->handle = at::cuda::getCurrentCUDABlasLtHandle();
        check(cublasLtMatmulDescCreate(&plan->matmul, compute, scale));
        const auto op = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
        check(cublasLtMatmulDescSetAttribute(plan->matmul, CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op)));
        check(cublasLtMatrixLayoutCreate(&plan->a, bf16, m, k, lda));
        check(cublasLtMatrixLayoutCreate(&plan->b, bf16, trans_b ? n : k, trans_b ? k : n, ldb));
        check(cublasLtMatrixLayoutCreate(&plan->c, bf16, m, n, ldc));
        plan->workspace_limit = workspace_limit;
        plan->enumerate(alignment, heuristic_limit, k);
        return plan.release();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "cuBLASLt plan creation failed: %s\n", error.what());
        return nullptr;
    }
}

extern "C" void head_lt_destroy(Plan* plan) { delete plan; }

extern "C" void head_lt_counts(const Plan* plan, int* counts) {
    counts[0] = plan->supported_ids;
    counts[1] = plan->validation_attempts;
    counts[2] = plan->candidates.size();
}

extern "C" float head_lt_info(const Plan* plan, int i, int* config) {
    const auto& candidate = plan->candidates[i];
    std::copy(candidate.config.begin(), candidate.config.end(), config);
    config[9] = candidate.heuristic_rank;
    return candidate.waves;
}

extern "C" size_t head_lt_workspace_size(const Plan* plan, int i) {
    return plan->candidates[i].workspace_size;
}

extern "C" int head_lt_run(const Plan* plan, int i, const void* a, const void* b, void* c, cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const auto& candidate = plan->candidates[i];
    void* workspace = candidate.workspace_size ? head_lt_inherited_workspace() : nullptr;
    if (candidate.workspace_size && !workspace) return CUBLAS_STATUS_ALLOC_FAILED;
    return cublasLtMatmul(plan->handle, plan->matmul, &alpha, a, plan->a, b, plan->b, &beta, c, plan->c, c,
                         plan->c, &candidate.algo, workspace, candidate.workspace_size, stream);
}
