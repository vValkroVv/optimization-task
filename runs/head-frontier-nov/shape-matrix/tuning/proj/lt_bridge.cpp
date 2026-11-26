// Reuse the pinned experiment's cuBLASLt algorithm search and ATen workspace ABI.
#include "../../../../fusion-followup/head_lt.cpp"

namespace {
void set_order(cublasLtMatrixLayout_t layout, bool row_major) {
    if (!row_major) return;
    const auto order = CUBLASLT_ORDER_ROW;
    check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
}

void set_batch(cublasLtMatrixLayout_t layout, int32_t count, int64_t stride) {
    if (count == 1) return;
    check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)));
    check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                          &stride, sizeof(stride)));
}
}

extern "C" Plan* lt_matrix_create(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
                                  int trans_a, int trans_b, int row_a, int row_b, int row_c,
                                  int32_t batch_count, int64_t step_a, int64_t step_b, int64_t step_c,
                                  uint32_t alignment, int heuristic_limit, size_t workspace_limit,
                                  bool exhaustive) {
    try {
        auto plan = std::make_unique<Plan>();
        plan->handle = at::cuda::getCurrentCUDABlasLtHandle();
        check(cublasLtMatmulDescCreate(&plan->matmul, compute, scale));
        const auto op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        const auto op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
        check(cublasLtMatmulDescSetAttribute(plan->matmul, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
        check(cublasLtMatmulDescSetAttribute(plan->matmul, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
        check(cublasLtMatrixLayoutCreate(&plan->a, bf16, trans_a ? k : m, trans_a ? m : k, lda));
        check(cublasLtMatrixLayoutCreate(&plan->b, bf16, trans_b ? n : k, trans_b ? k : n, ldb));
        check(cublasLtMatrixLayoutCreate(&plan->c, bf16, m, n, ldc));
        set_order(plan->a, row_a);
        set_order(plan->b, row_b);
        set_order(plan->c, row_c);
        set_batch(plan->a, batch_count, step_a);
        set_batch(plan->b, batch_count, step_b);
        set_batch(plan->c, batch_count, step_c);
        plan->workspace_limit = workspace_limit;
        if (exhaustive) {
            plan->enumerate(alignment, heuristic_limit, k);
        } else {
            check(cublasLtMatmulPreferenceCreate(&plan->preference));
            uint32_t mask = CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
            check(cublasLtMatmulPreferenceSetAttribute(plan->preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                       &mask, sizeof(mask)));
            for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                              CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
                check(cublasLtMatmulPreferenceSetAttribute(plan->preference, attr, &alignment, sizeof(alignment)));
            }
            plan->heuristic(0, heuristic_limit);
            if (workspace_limit) plan->heuristic(workspace_limit, heuristic_limit);
        }
        return plan.release();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "cuBLASLt matrix plan creation failed: %s\n", error.what());
        return nullptr;
    }
}
