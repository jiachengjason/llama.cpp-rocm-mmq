#include "mxfp4-f32.cuh"

#if defined(GGML_USE_HIP)

static constexpr int MXFP4_F32_TILE_X = 16;
static constexpr int MXFP4_F32_TILE_Y = 16;

static bool ggml_hip_mxfp4_f32_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_HIP_MXFP4_F32");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return enabled;
}

static __device__ __forceinline__ float ggml_cuda_mxfp4_scale(const block_mxfp4 & x) {
    return ggml_cuda_e8m0_to_fp32(x.e) * 0.5f;
}

static __device__ __forceinline__ float ggml_cuda_dot_mxfp4_f32(
        const block_mxfp4 * __restrict__ x, const float * __restrict__ y, const int64_t nblocks) {
    float sum = 0.0f;

    for (int64_t ib = 0; ib < nblocks; ++ib) {
        const block_mxfp4 xb = x[ib];
        const float d = ggml_cuda_mxfp4_scale(xb);
        const float * yb = y + ib*QK_MXFP4;

#pragma unroll
        for (int iq = 0; iq < QK_MXFP4/2; ++iq) {
            const uint8_t q = xb.qs[iq];
            sum += d * kvalues_mxfp4[q & 0x0f] * yb[iq];
            sum += d * kvalues_mxfp4[q >> 4]    * yb[iq + QK_MXFP4/2];
        }
    }

    return sum;
}

static __global__ void mul_mat_mxfp4_f32(
        const block_mxfp4 * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_y,
        const int64_t nchannels_x, const int64_t nchannels_y,
        const int64_t nsamples_x, const int64_t nsamples_y,
        const int64_t stride_row_x, const int64_t stride_channel_x, const int64_t stride_sample_x,
        const int64_t stride_col_y, const int64_t stride_channel_y, const int64_t stride_sample_y,
        const int64_t stride_col_dst, const int64_t stride_channel_dst, const int64_t stride_sample_dst) {
    const int64_t row = int64_t(blockIdx.x)*MXFP4_F32_TILE_Y + threadIdx.y;
    const int64_t col = int64_t(blockIdx.y)*MXFP4_F32_TILE_X + threadIdx.x;

    if (row >= nrows_x || col >= ncols_y) {
        return;
    }

    const int64_t sample_y  = blockIdx.z / nchannels_y;
    const int64_t channel_y = blockIdx.z - sample_y*nchannels_y;

    const int64_t channel_ratio = nchannels_y / nchannels_x;
    const int64_t sample_ratio  = nsamples_y  / nsamples_x;

    const int64_t channel_x = channel_y / channel_ratio;
    const int64_t sample_x  = sample_y  / sample_ratio;

    const int64_t nblocks = ncols_x / QK_MXFP4;

    const block_mxfp4 * x_row = x + sample_x*stride_sample_x + channel_x*stride_channel_x + row*stride_row_x;
    const float       * y_col = y + sample_y*stride_sample_y + channel_y*stride_channel_y + col*stride_col_y;
    float             * d_col = dst + sample_y*stride_sample_dst + channel_y*stride_channel_dst + col*stride_col_dst;

    d_col[row] = ggml_cuda_dot_mxfp4_f32(x_row, y_col, nblocks);
}

static __global__ void mul_mat_id_mxfp4_f32(
        const block_mxfp4 * __restrict__ x, const float * __restrict__ y, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_y,
        const int64_t n_experts, const int64_t n_expert_used, const int64_t n_tokens,
        const int64_t nsamples_x, const int64_t nsamples_y,
        const int64_t stride_row_x, const int64_t stride_channel_x, const int64_t stride_sample_x,
        const int64_t stride_col_y, const int64_t stride_token_y, const int64_t stride_sample_y,
        const int64_t stride_slot_dst, const int64_t stride_token_dst, const int64_t stride_sample_dst,
        const int64_t stride_slot_ids, const int64_t stride_token_ids) {
    const int64_t row = int64_t(blockIdx.x)*MXFP4_F32_TILE_Y + threadIdx.y;
    const int64_t col = int64_t(blockIdx.y)*MXFP4_F32_TILE_X + threadIdx.x;

    if (row >= nrows_x || col >= n_expert_used*n_tokens) {
        return;
    }

    const int64_t token = col / n_expert_used;
    const int64_t slot  = col - token*n_expert_used;

    const int32_t expert = ids[token*stride_token_ids + slot*stride_slot_ids];
    if (expert < 0 || expert >= n_experts) {
        return;
    }

    const int64_t sample_y = blockIdx.z;
    const int64_t sample_ratio = nsamples_y / nsamples_x;
    const int64_t sample_x = sample_y / sample_ratio;
    const int64_t nblocks = ncols_x / QK_MXFP4;

    const block_mxfp4 * x_row = x + sample_x*stride_sample_x + int64_t(expert)*stride_channel_x + row*stride_row_x;
    const float       * y_col = y + sample_y*stride_sample_y + token*stride_token_y + (slot % ncols_y)*stride_col_y;
    float             * d_col = dst + sample_y*stride_sample_dst + token*stride_token_dst + slot*stride_slot_dst;

    d_col[row] = ggml_cuda_dot_mxfp4_f32(x_row, y_col, nblocks);
}

#endif // defined(GGML_USE_HIP)

bool ggml_cuda_should_use_mxfp4_f32(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    if (!ggml_hip_mxfp4_f32_enabled()) {
        return false;
    }

    if (src0->type != GGML_TYPE_MXFP4 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (ids && ids->type != GGML_TYPE_I32) {
        return false;
    }

    if (src0->nb[0] != ggml_type_size(src0->type) ||
        src1->nb[0] != ggml_type_size(src1->type) ||
        dst->nb[0]  != ggml_type_size(dst->type)) {
        return false;
    }

    if (src0->ne[0] != src1->ne[0] || src0->ne[0] % QK_MXFP4 != 0) {
        return false;
    }

    if (!ids) {
        if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
            return false;
        }
        return true;
    }

    if (dst->ne[1] != ids->ne[0] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        return false;
    }

    if (src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }

    return true;
#else
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    return false;
#endif // defined(GGML_USE_HIP)
}

void ggml_cuda_mul_mat_mxfp4_f32(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    GGML_ASSERT(ggml_cuda_should_use_mxfp4_f32(src0, src1, ids, dst));

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(nb00 == ts_src0);
    GGML_ASSERT(nb10 == ts_src1);
    GGML_ASSERT(nb0  == ts_dst);

    const block_mxfp4 * src0_d = (const block_mxfp4 *) src0->data;
    const float       * src1_d = (const float       *) src1->data;
    float             * dst_d  = (float             *) dst->data;

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s03 = src0->nb[3] / ts_src0;

    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s13 = src1->nb[3] / ts_src1;

    const int64_t s1 = dst->nb[1] / ts_dst;
    const int64_t s2 = dst->nb[2] / ts_dst;
    const int64_t s3 = dst->nb[3] / ts_dst;

    const dim3 block_nums(
        (ne01 + MXFP4_F32_TILE_Y - 1) / MXFP4_F32_TILE_Y,
        ((ids ? ne12*ids->ne[0] : ne11) + MXFP4_F32_TILE_X - 1) / MXFP4_F32_TILE_X,
        ids ? ne13 : ne12*ne13);
    const dim3 block_dims(MXFP4_F32_TILE_X, MXFP4_F32_TILE_Y, 1);

    if (ids) {
        GGML_ASSERT(ids->nb[0] == ggml_type_size(ids->type));
        const int64_t si0 = ids->nb[0] / ggml_type_size(ids->type);
        const int64_t si1 = ids->nb[1] / ggml_type_size(ids->type);

        mul_mat_id_mxfp4_f32<<<block_nums, block_dims, 0, stream>>>(
            src0_d, src1_d, (const int32_t *) ids->data, dst_d,
            ne00, ne01, ne11, ne02, ids->ne[0], ne12,
            ne03, ne13,
            s01, s02, s03,
            s11, s12, s13,
            s1, s2, s3,
            si0, si1);
    } else {
        mul_mat_mxfp4_f32<<<block_nums, block_dims, 0, stream>>>(
            src0_d, src1_d, dst_d,
            ne00, ne01, ne11,
            ne02, ne12,
            ne03, ne13,
            s01, s02, s03,
            s11, s12, s13,
            s1, s2, s3);
    }

    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    GGML_ABORT("MXFP4 x F32 direct matmul is only implemented for HIP");
#endif // defined(GGML_USE_HIP)
}
