#pragma once

#include "common.cuh"

bool ggml_cuda_should_use_mxfp4_f32(
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);

void ggml_cuda_mul_mat_mxfp4_f32(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    ggml_tensor * dst);
