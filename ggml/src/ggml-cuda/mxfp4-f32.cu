#include "mxfp4-f32.cuh"
#include "unary.cuh"

#if defined(GGML_USE_HIP)

static constexpr int MXFP4_F32_MM_TILE_N = 4;
static constexpr int MXFP4_F32_MM_TILE_M = 16;
static constexpr int MXFP4_F32_MM_K_LANES = 4;
static constexpr int MXFP4_F32_MM_NTHREADS = MXFP4_F32_MM_TILE_N*MXFP4_F32_MM_TILE_M*MXFP4_F32_MM_K_LANES;

static constexpr int MXFP4_F32_ID_TILE_N = 1;
static constexpr int MXFP4_F32_ID_TILE_M = 16;
static constexpr int MXFP4_F32_ID_K_LANES = 16;
static constexpr int MXFP4_F32_ID_NTHREADS = MXFP4_F32_ID_TILE_N*MXFP4_F32_ID_TILE_M*MXFP4_F32_ID_K_LANES;

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

static __device__ __forceinline__ float ggml_cuda_mxfp4_value(const block_mxfp4 & x, const int ik) {
    const uint8_t q = x.qs[ik & (QK_MXFP4/2 - 1)];
    const int8_t qv = ik < QK_MXFP4/2 ? kvalues_mxfp4[q & 0x0f] : kvalues_mxfp4[q >> 4];
    return ggml_cuda_mxfp4_scale(x) * qv;
}

static __device__ __forceinline__ float ggml_cuda_mxfp4_f32_apply_glu(
        const float value, const float gate, const ggml_glu_op glu_op) {
    switch (glu_op) {
        case GGML_GLU_OP_SWIGLU:
            return value * ggml_cuda_op_silu_single(gate);
        case GGML_GLU_OP_GEGLU:
            return value * ggml_cuda_op_gelu_single(gate);
        case GGML_GLU_OP_SWIGLU_OAI:
            return ggml_cuda_op_swiglu_oai_single(gate, value);
        default:
            return value * gate;
    }
}

template <bool has_gate>
static __global__ void mul_mat_mxfp4_f32(
        const block_mxfp4 * __restrict__ x, const float * __restrict__ y,
        const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_y,
        const int64_t nchannels_x, const int64_t nchannels_y,
        const int64_t nsamples_x, const int64_t nsamples_y,
        const int64_t stride_row_x, const int64_t stride_channel_x, const int64_t stride_sample_x,
        const int64_t stride_col_y, const int64_t stride_channel_y, const int64_t stride_sample_y,
        const int64_t stride_col_dst, const int64_t stride_channel_dst, const int64_t stride_sample_dst) {
    __shared__ float tile_x[MXFP4_F32_MM_K_LANES][MXFP4_F32_MM_TILE_M][QK_MXFP4];
    [[maybe_unused]] __shared__ float tile_g[has_gate ? MXFP4_F32_MM_K_LANES : 1][has_gate ? MXFP4_F32_MM_TILE_M : 1][has_gate ? QK_MXFP4 : 1];
    __shared__ float tile_y[MXFP4_F32_MM_K_LANES][MXFP4_F32_MM_TILE_N][QK_MXFP4];
    __shared__ float partial[MXFP4_F32_MM_K_LANES][MXFP4_F32_MM_TILE_M][MXFP4_F32_MM_TILE_N];
    [[maybe_unused]] __shared__ float partial_g[has_gate ? MXFP4_F32_MM_K_LANES : 1][has_gate ? MXFP4_F32_MM_TILE_M : 1][has_gate ? MXFP4_F32_MM_TILE_N : 1];

    const int tid = (threadIdx.z*MXFP4_F32_MM_TILE_M + threadIdx.y)*MXFP4_F32_MM_TILE_N + threadIdx.x;
    const int lane = threadIdx.z;
    const int row_in_tile = threadIdx.y;
    const int col_in_tile = threadIdx.x;

    const int64_t row = int64_t(blockIdx.x)*MXFP4_F32_MM_TILE_M + row_in_tile;
    const int64_t col = int64_t(blockIdx.y)*MXFP4_F32_MM_TILE_N + col_in_tile;

    const int64_t sample_y  = blockIdx.z / nchannels_y;
    const int64_t channel_y = blockIdx.z - sample_y*nchannels_y;

    const int64_t channel_ratio = nchannels_y / nchannels_x;
    const int64_t sample_ratio  = nsamples_y  / nsamples_x;

    const int64_t channel_x = channel_y / channel_ratio;
    const int64_t sample_x  = sample_y  / sample_ratio;

    const int64_t nblocks = ncols_x / QK_MXFP4;

    const block_mxfp4 * x_base = x + sample_x*stride_sample_x + channel_x*stride_channel_x;
    [[maybe_unused]] const block_mxfp4 * g_base = has_gate ? (const block_mxfp4 *) fusion.gate + sample_x*stride_sample_x + channel_x*stride_channel_x : nullptr;
    const float       * y_base = y + sample_y*stride_sample_y + channel_y*stride_channel_y;

    float sum = 0.0f;
    [[maybe_unused]] float sum_g = 0.0f;

    for (int64_t kb0 = 0; kb0 < nblocks; kb0 += MXFP4_F32_MM_K_LANES) {
        for (int i = tid; i < MXFP4_F32_MM_K_LANES*MXFP4_F32_MM_TILE_M*QK_MXFP4; i += MXFP4_F32_MM_NTHREADS) {
            const int ik = i % QK_MXFP4;
            const int ir = (i / QK_MXFP4) % MXFP4_F32_MM_TILE_M;
            const int il = i / (QK_MXFP4*MXFP4_F32_MM_TILE_M);

            const int64_t row_cur = int64_t(blockIdx.x)*MXFP4_F32_MM_TILE_M + ir;
            const int64_t kb = kb0 + il;

            float v = 0.0f;
            if (row_cur < nrows_x && kb < nblocks) {
                const block_mxfp4 xb = x_base[row_cur*stride_row_x + kb];
                v = ggml_cuda_mxfp4_value(xb, ik);
            }
            tile_x[il][ir][ik] = v;

            if constexpr (has_gate) {
                v = 0.0f;
                if (row_cur < nrows_x && kb < nblocks) {
                    const block_mxfp4 gb = g_base[row_cur*stride_row_x + kb];
                    v = ggml_cuda_mxfp4_value(gb, ik);
                }
                tile_g[il][ir][ik] = v;
            }
        }

        for (int i = tid; i < MXFP4_F32_MM_K_LANES*MXFP4_F32_MM_TILE_N*QK_MXFP4; i += MXFP4_F32_MM_NTHREADS) {
            const int ik = i % QK_MXFP4;
            const int ic = (i / QK_MXFP4) % MXFP4_F32_MM_TILE_N;
            const int il = i / (QK_MXFP4*MXFP4_F32_MM_TILE_N);

            const int64_t col_cur = int64_t(blockIdx.y)*MXFP4_F32_MM_TILE_N + ic;
            const int64_t kb = kb0 + il;

            float v = 0.0f;
            if (col_cur < ncols_y && kb < nblocks) {
                v = y_base[col_cur*stride_col_y + kb*QK_MXFP4 + ik];
            }
            tile_y[il][ic][ik] = v;
        }

        __syncthreads();

        if (row < nrows_x && col < ncols_y) {
#pragma unroll
            for (int ik = 0; ik < QK_MXFP4; ++ik) {
                const float yv = tile_y[lane][col_in_tile][ik];
                sum += tile_x[lane][row_in_tile][ik] * yv;
                if constexpr (has_gate) {
                    sum_g += tile_g[lane][row_in_tile][ik] * yv;
                }
            }
        }

        __syncthreads();
    }

    partial[lane][row_in_tile][col_in_tile] = sum;
    if constexpr (has_gate) {
        partial_g[lane][row_in_tile][col_in_tile] = sum_g;
    }
    __syncthreads();

    if (lane == 0 && row < nrows_x && col < ncols_y) {
        float total = 0.0f;
        [[maybe_unused]] float total_g = 0.0f;
#pragma unroll
        for (int il = 0; il < MXFP4_F32_MM_K_LANES; ++il) {
            total += partial[il][row_in_tile][col_in_tile];
            if constexpr (has_gate) {
                total_g += partial_g[il][row_in_tile][col_in_tile];
            }
        }

        float * d_col = dst + sample_y*stride_sample_dst + channel_y*stride_channel_dst + col*stride_col_dst;
        if (fusion.x_bias) {
            const float * x_bias = (const float *) fusion.x_bias + sample_y*stride_sample_dst + channel_y*stride_channel_dst + col*stride_col_dst;
            total += x_bias[row];
        }
        if constexpr (has_gate) {
            if (fusion.gate_bias) {
                const float * gate_bias = (const float *) fusion.gate_bias + sample_y*stride_sample_dst + channel_y*stride_channel_dst + col*stride_col_dst;
                total_g += gate_bias[row];
            }
            total = ggml_cuda_mxfp4_f32_apply_glu(total, total_g, fusion.glu_op);
        }
        d_col[row] = total;
    }
}

template <bool has_gate>
static __global__ void mul_mat_id_mxfp4_f32(
        const block_mxfp4 * __restrict__ x, const float * __restrict__ y, const int32_t * __restrict__ ids,
        const ggml_cuda_mm_fusion_args_device fusion,
        float * __restrict__ dst,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_y,
        const int64_t n_experts, const int64_t n_expert_used, const int64_t n_tokens,
        const int64_t nsamples_x, const int64_t nsamples_y,
        const int64_t stride_row_x, const int64_t stride_channel_x, const int64_t stride_sample_x,
        const int64_t stride_col_y, const int64_t stride_token_y, const int64_t stride_sample_y,
        const int64_t stride_slot_dst, const int64_t stride_token_dst, const int64_t stride_sample_dst,
        const int64_t stride_slot_ids, const int64_t stride_token_ids) {
    __shared__ float tile_y[MXFP4_F32_ID_K_LANES][MXFP4_F32_ID_TILE_N][QK_MXFP4];
    __shared__ float partial[MXFP4_F32_ID_K_LANES][MXFP4_F32_ID_TILE_M][MXFP4_F32_ID_TILE_N];
    [[maybe_unused]] __shared__ float partial_g[has_gate ? MXFP4_F32_ID_K_LANES : 1][has_gate ? MXFP4_F32_ID_TILE_M : 1][has_gate ? MXFP4_F32_ID_TILE_N : 1];
    __shared__ int32_t expert_tile[MXFP4_F32_ID_TILE_N];
    __shared__ int token_tile[MXFP4_F32_ID_TILE_N];
    __shared__ int slot_tile[MXFP4_F32_ID_TILE_N];

    const int tid = (threadIdx.z*MXFP4_F32_ID_TILE_M + threadIdx.y)*MXFP4_F32_ID_TILE_N + threadIdx.x;
    const int lane = threadIdx.z;
    const int row_in_tile = threadIdx.y;
    const int col_in_tile = threadIdx.x;

    const int64_t row = int64_t(blockIdx.x)*MXFP4_F32_ID_TILE_M + row_in_tile;
    const int64_t col = int64_t(blockIdx.y)*MXFP4_F32_ID_TILE_N + col_in_tile;

    if (tid < MXFP4_F32_ID_TILE_N) {
        const int64_t col_cur = int64_t(blockIdx.y)*MXFP4_F32_ID_TILE_N + tid;
        if (col_cur < n_expert_used*n_tokens) {
            const int token = col_cur / n_expert_used;
            const int slot = col_cur - int64_t(token)*n_expert_used;
            token_tile[tid] = token;
            slot_tile[tid] = slot;
            expert_tile[tid] = ids[int64_t(token)*stride_token_ids + int64_t(slot)*stride_slot_ids];
        } else {
            token_tile[tid] = 0;
            slot_tile[tid] = 0;
            expert_tile[tid] = -1;
        }
    }

    __syncthreads();

    const int token = token_tile[col_in_tile];
    const int slot  = slot_tile[col_in_tile];
    const int32_t expert = expert_tile[col_in_tile];

    const int64_t sample_y = blockIdx.z;
    const int64_t sample_ratio = nsamples_y / nsamples_x;
    const int64_t sample_x = sample_y / sample_ratio;
    const int64_t nblocks = ncols_x / QK_MXFP4;

    const block_mxfp4 * x_base = x + sample_x*stride_sample_x;
    [[maybe_unused]] const block_mxfp4 * g_base = has_gate ? (const block_mxfp4 *) fusion.gate + sample_x*stride_sample_x : nullptr;
    const float       * y_base = y + sample_y*stride_sample_y;

    float sum = 0.0f;
    [[maybe_unused]] float sum_g = 0.0f;

    for (int64_t kb0 = 0; kb0 < nblocks; kb0 += MXFP4_F32_ID_K_LANES) {
        for (int i = tid; i < MXFP4_F32_ID_K_LANES*MXFP4_F32_ID_TILE_N*QK_MXFP4; i += MXFP4_F32_ID_NTHREADS) {
            const int ik = i % QK_MXFP4;
            const int ic = (i / QK_MXFP4) % MXFP4_F32_ID_TILE_N;
            const int il = i / (QK_MXFP4*MXFP4_F32_ID_TILE_N);

            const int token_cur = token_tile[ic];
            const int slot_cur = slot_tile[ic];
            const int64_t kb = kb0 + il;

            float v = 0.0f;
            if (expert_tile[ic] >= 0 && kb < nblocks) {
                v = y_base[int64_t(token_cur)*stride_token_y + int64_t(slot_cur % ncols_y)*stride_col_y + kb*QK_MXFP4 + ik];
            }
            tile_y[il][ic][ik] = v;
        }

        __syncthreads();

        const int64_t kb = kb0 + lane;
        if (row < nrows_x && col < n_expert_used*n_tokens && expert >= 0 && expert < n_experts && kb < nblocks) {
            const block_mxfp4 xb = x_base[int64_t(expert)*stride_channel_x + row*stride_row_x + kb];
            const float d = ggml_cuda_mxfp4_scale(xb);
            [[maybe_unused]] block_mxfp4 gb;
            [[maybe_unused]] float dg = 0.0f;
            if constexpr (has_gate) {
                gb = g_base[int64_t(expert)*stride_channel_x + row*stride_row_x + kb];
                dg = ggml_cuda_mxfp4_scale(gb);
            }

#pragma unroll
            for (int iq = 0; iq < QK_MXFP4/2; ++iq) {
                const uint8_t q = xb.qs[iq];
                sum += d * kvalues_mxfp4[q & 0x0f] * tile_y[lane][col_in_tile][iq];
                sum += d * kvalues_mxfp4[q >> 4]    * tile_y[lane][col_in_tile][iq + QK_MXFP4/2];

                if constexpr (has_gate) {
                    const uint8_t qg = gb.qs[iq];
                    sum_g += dg * kvalues_mxfp4[qg & 0x0f] * tile_y[lane][col_in_tile][iq];
                    sum_g += dg * kvalues_mxfp4[qg >> 4]    * tile_y[lane][col_in_tile][iq + QK_MXFP4/2];
                }
            }
        }

        __syncthreads();
    }

    partial[lane][row_in_tile][col_in_tile] = sum;
    if constexpr (has_gate) {
        partial_g[lane][row_in_tile][col_in_tile] = sum_g;
    }
    __syncthreads();

    if (lane == 0 && row < nrows_x && col < n_expert_used*n_tokens && expert >= 0 && expert < n_experts) {
        float total = 0.0f;
        [[maybe_unused]] float total_g = 0.0f;
#pragma unroll
        for (int il = 0; il < MXFP4_F32_ID_K_LANES; ++il) {
            total += partial[il][row_in_tile][col_in_tile];
            if constexpr (has_gate) {
                total_g += partial_g[il][row_in_tile][col_in_tile];
            }
        }

        float * d_col = dst + sample_y*stride_sample_dst + int64_t(token)*stride_token_dst + int64_t(slot)*stride_slot_dst;
        if (fusion.x_bias) {
            const float * x_bias = (const float *) fusion.x_bias + sample_y*stride_sample_dst + int64_t(token)*stride_token_dst + int64_t(expert)*stride_slot_dst;
            total += x_bias[row];
        }
        if constexpr (has_gate) {
            if (fusion.gate_bias) {
                const float * gate_bias = (const float *) fusion.gate_bias + sample_y*stride_sample_dst + int64_t(token)*stride_token_dst + int64_t(expert)*stride_slot_dst;
                total_g += gate_bias[row];
            }
            total = ggml_cuda_mxfp4_f32_apply_glu(total, total_g, fusion.glu_op);
        }
        d_col[row] = total;
    }
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
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
#if defined(GGML_USE_HIP)
    const bool has_fusion = fusion && (fusion->x_bias || fusion->gate || fusion->gate_bias);
    if (!has_fusion) {
        GGML_ASSERT(ggml_cuda_should_use_mxfp4_f32(src0, src1, ids, dst));
    } else {
        GGML_ASSERT(src0->type == GGML_TYPE_MXFP4);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type  == GGML_TYPE_F32);
        GGML_ASSERT(!ids || ids->type == GGML_TYPE_I32);
        GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
        GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
        GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));
        GGML_ASSERT(src0->ne[0] == src1->ne[0]);
        GGML_ASSERT(src0->ne[0] % QK_MXFP4 == 0);
        GGML_ASSERT(!ids || dst->ne[2] == 1);
        GGML_ASSERT( ids || dst->ne[1] == 1);
    }

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
    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (has_fusion) {
        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s03 = src0->nb[3] / ts_src0;

    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s13 = src1->nb[3] / ts_src1;

    const int64_t s1 = dst->nb[1] / ts_dst;
    const int64_t s2 = dst->nb[2] / ts_dst;
    const int64_t s3 = dst->nb[3] / ts_dst;

    if (ids) {
        GGML_ASSERT(ids->nb[0] == ggml_type_size(ids->type));
        const int64_t si0 = ids->nb[0] / ggml_type_size(ids->type);
        const int64_t si1 = ids->nb[1] / ggml_type_size(ids->type);
        const dim3 block_nums(
            (ne01 + MXFP4_F32_ID_TILE_M - 1) / MXFP4_F32_ID_TILE_M,
            (ne12*ids->ne[0] + MXFP4_F32_ID_TILE_N - 1) / MXFP4_F32_ID_TILE_N,
            ne13);
        const dim3 block_dims(MXFP4_F32_ID_TILE_N, MXFP4_F32_ID_TILE_M, MXFP4_F32_ID_K_LANES);

        if (fusion_local.gate) {
            mul_mat_id_mxfp4_f32<true><<<block_nums, block_dims, 0, stream>>>(
                src0_d, src1_d, (const int32_t *) ids->data, fusion_local, dst_d,
                ne00, ne01, ne11, ne02, ids->ne[0], ne12,
                ne03, ne13,
                s01, s02, s03,
                s11, s12, s13,
                s1, s2, s3,
                si0, si1);
        } else {
            mul_mat_id_mxfp4_f32<false><<<block_nums, block_dims, 0, stream>>>(
                src0_d, src1_d, (const int32_t *) ids->data, fusion_local, dst_d,
                ne00, ne01, ne11, ne02, ids->ne[0], ne12,
                ne03, ne13,
                s01, s02, s03,
                s11, s12, s13,
                s1, s2, s3,
                si0, si1);
        }
    } else {
        const dim3 block_nums(
            (ne01 + MXFP4_F32_MM_TILE_M - 1) / MXFP4_F32_MM_TILE_M,
            (ne11 + MXFP4_F32_MM_TILE_N - 1) / MXFP4_F32_MM_TILE_N,
            ne12*ne13);
        const dim3 block_dims(MXFP4_F32_MM_TILE_N, MXFP4_F32_MM_TILE_M, MXFP4_F32_MM_K_LANES);

        if (fusion_local.gate) {
            mul_mat_mxfp4_f32<true><<<block_nums, block_dims, 0, stream>>>(
                src0_d, src1_d, fusion_local, dst_d,
                ne00, ne01, ne11,
                ne02, ne12,
                ne03, ne13,
                s01, s02, s03,
                s11, s12, s13,
                s1, s2, s3);
        } else {
            mul_mat_mxfp4_f32<false><<<block_nums, block_dims, 0, stream>>>(
                src0_d, src1_d, fusion_local, dst_d,
                ne00, ne01, ne11,
                ne02, ne12,
                ne03, ne13,
                s01, s02, s03,
                s11, s12, s13,
                s1, s2, s3);
        }
    }

    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    GGML_UNUSED(fusion);
    GGML_ABORT("MXFP4 x F32 direct matmul is only implemented for HIP");
#endif // defined(GGML_USE_HIP)
}
