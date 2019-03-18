#include <stdio.h>
#include "fuzzy.h"

#define CACHE_LINE_SIZE 128
#define T_NUM 11
#define IMPL(a, b) (1 - (a) + (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))



__device__ unsigned int lock = 0;

__global__
void predict_gpu_kernel(const float* fsets_table[],  const unsigned* fsets_lens, const unsigned* fsets_dims,
                        const float* a0, const unsigned char* a_indices, const unsigned char* b_indices, volatile float* b0,
                        unsigned N, unsigned n)
{
    // grid_group g = this_grid();

    // NOTE(sergey): Block size must be set to multiple of a warp size by host caller.
    unsigned warp_multiple_len = blockDim.x;
    unsigned attr_index = blockIdx.x;
    unsigned i, j, k, ti;

    if (blockIdx.x == n) {
        float min = 0;
        for (k = 0; k < N; ++k) {
            // g.sync();
            __threadfence();

            float max = b0[threadIdx.x];
            for (i = 1; i < n; ++i) {
                max = fmaxf(max, b0[i * warp_multiple_len + threadIdx.x]);
            }
            min = fminf(min, max);
        }
        b0[threadIdx.x] = min;
    } else {
        unsigned a_len = fsets_lens[attr_index];
        unsigned a_n = fsets_dims[attr_index];
        unsigned b_len = fsets_lens[n];
        unsigned b_n = fsets_dims[n];

        // NOTE(sergey): Block size should be warp divisible to create appropriate shared memory layout
        // that allows omit an if-statement when working with these data.
        extern __shared__ float shared_buffer[];

        // float a0_local[warp_multiple_len + 1]; // a_n
        // float a_sets_local[a_len][warp_multiple_len + 1]; // a_n
        // float b_sets_local[b_len][warp_multiple_len + 1]; // b_n
        // float ftp[warp_multiple_len + 1]; // T_NUM
        // float tmp_b0[warp_multiple_len + 1]; // b_n
        float* a0_local = shared_buffer;
        float* a_sets_local = a0_local + warp_multiple_len + 1;
        float* b_sets_local = a_sets_local + a_len * (warp_multiple_len + 1);
        float* ftp = b_sets_local + b_len * (warp_multiple_len + 1);
        float* tmp_b0 = ftp + warp_multiple_len + 1;

        // TODO(sergey): Decide whether a0 rows
        a0_local[threadIdx.x] = a0[attr_index * warp_multiple_len + threadIdx.x]; // TODO(sergey): a0 must be table of pointers in this case.
        if (threadIdx.x == 0) a0_local[a_n] = a0_local[a_n - 1];
        for (i = 0; i < a_len; ++i) {
            a_sets_local[i * (warp_multiple_len + 1) + threadIdx.x] = fsets_table[attr_index][i * warp_multiple_len + threadIdx.x];
            if (threadIdx.x == 0) a_sets_local[i * (warp_multiple_len + 1) + a_n] = a_sets_local[i * (warp_multiple_len + 1) + a_n - 1];
        }
        for (i = 0; i < b_len; ++i) {
            b_sets_local[i * (warp_multiple_len + 1) + threadIdx.x] = fsets_table[n][i * warp_multiple_len + threadIdx.x];
            if (threadIdx.x == 0) b_sets_local[i * (warp_multiple_len + 1) + b_n] = b_sets_local[i * (warp_multiple_len + 1) + b_n - 1];
        }
        b0[n * warp_multiple_len + threadIdx.x] = 1.f;

        for (k = 0; k < N; ++k) {
            float* a = a_sets_local + a_indices[attr_index * N + k] * (warp_multiple_len + 1);
            float* b = b_sets_local + b_indices[k] * (warp_multiple_len + 1);
            ti = threadIdx.x;

            float t = (float) ti / (T_NUM - 1);

            // NOTE(sergey): I think computing ftp[threadIdx.x] for each threadIdx.x would be betten than
            // using a[threadIdx.x] and a[threadIdx.x]. Because in the last case we should use atomic operation
            // when updating ftp[j].
            ftp[ti] = 0.f;
            for (i = 0; i < a_n; ++i) {
                float a1 = a[i];
                float a2 = a[i + 1];
                if ((t - a1) * (a2 - t) >= 0) {
                    // float x1 = (float) i / a_n;
                    // float x2 = (float) (i + 1) / a_n;
                    // float x = x1 + (t - a1) * (x2 - x1) / (a2 - a1);
                    // float y = a0_local[i] + (x - x1) * (a0_local[i+1] - a0_local[i]) / (x2 - x1);
                    float y = a0_local[i] + (t - a1) * (a0_local[i+1] - a0_local[i]) / (a2 - a1);
                    // ftp[ti] = fmaxf(ftp[ti], y);
                    if (y > ftp[ti]) ftp[ti] = y;
                }
            }

            __syncthreads();

            tmp_b0[threadIdx.x] = 0.f;
            for (j = 0; j < T_NUM; ++j) {
                tmp_b0[threadIdx.x] = fmaxf(tmp_b0[threadIdx.x], fminf(ftp[j], IMPL((float)j / (T_NUM - 1), b[threadIdx.x])));
            }
            b0[attr_index * warp_multiple_len + threadIdx.x] = tmp_b0[threadIdx.x];
            // b0[n * warp_multiple_len + blockIdx.x] = attr_index;//tmp_b0[threadIdx.x];

            __threadfence();

            for (i = 1; i < 6;) {
                i <<= 1;
                if (blockIdx.x & i == i && blockIdx.x + i/2 < a_n) {
                    b0[attr_index * warp_multiple_len + threadIdx.x] =
                        fmaxf(b0[attr_index * warp_multiple_len + threadIdx.x],
                              b0[(attr_index + i) * warp_multiple_len + threadIdx.x]);
                }
                __threadfence();
                atomicInc(&lock, 1);
                __threadfence();
                while (atomicCAS(&lock, gridDim.x * blockDim.x, 0) != 0) continue;
                __threadfence();
                atomicDec(&lock, 1);
            }
            // if (blockIdx.x == 0) {
            //     b0[n * warp_multiple_len + threadIdx.x] =
            //         fminf(b0[n * warp_multiple_len + threadIdx.x], b0[threadIdx.x]);
            // }
            // g.sync();
        }
    }
}


extern "C"
void predict_gpu(const float** fsets[], const unsigned* fsets_lens, const unsigned* fsets_dims,
                 const float* a0[], const unsigned char* a[], const unsigned char* b, float* b0,
                 unsigned N, unsigned n)
{
    /*
     * GPU global memory layout:
     * +-----------------------------------+
     * | [META] Fuzzy sets lens + dims     |
     * +-----------------------------------+
     * | Fuzzy sets table (Ptrs to buffer) |
     * +-----------------------------------+
     * | Fuzzy sets buffer (data store)    |
     * +-----------------------------------+
     * | a0 (data store)                   |
     * +-----------------------------------+
     * | a (indices in the F-sets table)   |
     * +-----------------------------------+
     * | b (indices in the F-sets table)   |
     * +-----------------------------------+
     * | b0 (data store)                   |
     * +-----------------------------------+
     *
     * Note: N-th F-sets table entry corresponds to b's underling attribute.
     */

    unsigned i, j;
    unsigned block_sz = 0;

    for (i = 0; i < n + 1; ++i) block_sz = MAX(block_sz, fsets_dims[i]);
    block_sz = (block_sz + 31) & ~0x1F;

    unsigned *fsets_lens_d, *fsets_dims_d;
    float **fsets_table_d, *fsets_table[n + 1];
    float* fsets_buffer_d, *a0_d, *b0_d;
    unsigned char *a_d, *b_d;
    unsigned fsets_buffer_sz = 0, offset;

    for (i = 0; i < n + 1; ++i) fsets_buffer_sz += fsets_lens[i];
    fsets_buffer_sz *= block_sz;

    cudaMalloc(&fsets_lens_d, sizeof(unsigned[n + 1]));
    cudaMalloc(&fsets_dims_d, sizeof(unsigned[n + 1]));
    cudaMalloc(&fsets_table_d, sizeof(float*[n + 1]));
    cudaMalloc(&fsets_buffer_d, sizeof(float[fsets_buffer_sz]));
    cudaMalloc(&a0_d, sizeof(float[n][block_sz]));
    cudaMalloc(&a_d, sizeof(unsigned[n][N]));
    cudaMalloc(&b_d, sizeof(unsigned[N]));
    cudaMalloc(&b0_d, sizeof(float[n + 1][block_sz]));

    cudaMemset(fsets_buffer_d, 0, sizeof(float[fsets_buffer_sz]));
    cudaMemset(a0_d, 0, sizeof(float[n][block_sz]));
    offset = 0;
    for (i = 0; i < n + 1; ++i) {
        fsets_table[i] = fsets_buffer_d + offset;
        for (j = 0; j < fsets_lens[i]; ++j) {
            cudaMemcpy(fsets_table[i] + j * block_sz, fsets[i][j], sizeof(float[fsets_dims[i]]), cudaMemcpyHostToDevice);
        }
        offset += fsets_lens[i] * block_sz;
    }

    cudaMemcpy(fsets_lens_d, fsets_lens, sizeof(unsigned[n + 1]), cudaMemcpyHostToDevice);
    cudaMemcpy(fsets_dims_d, fsets_dims, sizeof(unsigned[n + 1]), cudaMemcpyHostToDevice);
    cudaMemcpy(fsets_table_d, fsets_table, sizeof(float*[n + 1]), cudaMemcpyHostToDevice);
    for (i = 0; i < n; ++i) {
        cudaMemcpy(a0_d + i * block_sz, a0[i], sizeof(float[fsets_dims[i]]), cudaMemcpyHostToDevice);
        cudaMemcpy(a_d + i * N, a[i], sizeof(unsigned char[N]), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(b_d, b, sizeof(unsigned char[N]), cudaMemcpyHostToDevice);
    cudaMemset(b0_d, 0, sizeof(float[n][fsets_dims[n]]));

    unsigned a_len_max = 0, b_len = fsets_lens[n];
    for (i = 0; i < n; ++i) a_len_max = MAX(a_len_max, fsets_lens[i]);

    unsigned shared_per_block_sz = sizeof(float[a_len_max + b_len + /* a0 + ftp + tmp_b0 */ 3][block_sz + 1]);

    predict_gpu_kernel<<<n, block_sz, shared_per_block_sz>>>((const float**)fsets_table_d, fsets_lens_d, fsets_dims_d, a0_d, a_d, b_d, b0_d, N, n);
    cudaDeviceSynchronize();

    for (i = 0; i < n+1; ++i) {
        cudaMemcpy(b0 + i * fsets_dims[n], b0_d + i * block_sz, sizeof(float[fsets_dims[n]]), cudaMemcpyDeviceToHost);
    }
    // cudaMemcpy(b0, b0_d /*+ n * block_sz*/, sizeof(float[fsets_dims[n]]), cudaMemcpyDeviceToHost);
    cudaFree(fsets_lens_d);
    cudaFree(fsets_dims_d);
    cudaFree(fsets_table_d);
    cudaFree(fsets_buffer_d);
    cudaFree(a0_d);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(b0_d);
}