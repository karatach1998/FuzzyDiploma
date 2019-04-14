#include <stdio.h>
#include "fuzzy.h"

#define CACHE_LINE_SIZE 128
#define T_NUM 11
#define IMPL(a, b) (1 - (a) + (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define WARP_SIZE 32
#define WARP_MULTIPLE(x) (((x) + (WARP_SIZE-1)) & ~(WARP_SIZE-1))

#define CU_HANDLE_ERROR(e) cuda_handle_error((e), __FILE__, __LINE__, #e)

static void cuda_handle_error(cudaError_t e, const char* file, int line, const char* src)
{
    if (e != cudaSuccess)
    {
        fprintf(stderr, "CUDA-ERROR: %s:%d: %s <%s>\n", file, line, cudaGetErrorString(e), src);
        switch (e) {
            case cudaErrorInvalidValue: fprintf(stderr, "InvalidValue\n"); break;
            case cudaErrorMemoryAllocation: fprintf(stderr, "MemoryAllocation\n"); break;
            case cudaErrorHostMemoryAlreadyRegistered: fprintf(stderr, "HostMemoryAlreadyRegistered\n"); break;
            case cudaErrorNotSupported: fprintf(stderr, "NotSupported\n"); break;
        }
        exit(EXIT_FAILURE);
    }
}

__global__
void compute_ftp_kernel(const float* fsets, const float* a0, const unsigned char* a_indices, float* ftp_buf,
                        unsigned attr_index, unsigned a_len, unsigned a_n, unsigned b_n, unsigned N, unsigned n)
{
    unsigned i, j, k;

    unsigned ftp_buf_entry_sz = WARP_MULTIPLE(MAX(T_NUM, b_n));

    // __shared__ float a_cache[a_len][a_n+1];
    // __shared__ float a0_cache[a_n+1];
    // __shared__ float ftp[T_NUM];

    extern __shared__ float cache[];

    float* a_cache = cache;
    float* a0_cache = a_cache + a_len * (a_n+1);
    float* ftp = a0_cache + (a_n+1);

    for (k = 0; k < a_len; ++k) {
        for (i = threadIdx.x; i < a_n; i += blockDim.x) a_cache[k * (a_n+1) + i] = fsets[k * a_n + i];
        if (threadIdx.x == 0) a_cache[k * (a_n+1) + a_n] = a_cache[k * (a_n+1) + a_n - 1];
    }
    for (i = threadIdx.x; i < a_n; i += blockDim.x) a0_cache[i] = a0[i];
    if (threadIdx.x == 0) a0_cache[a_n] = a0_cache[a_n-1];

    unsigned ti = threadIdx.x;
    float t = (float) ti / (T_NUM - 1);

    for (k = blockIdx.x; k < N; k += gridDim.x) {
        float* a = a_cache + a_indices[k] * (a_n+1);

        ftp[ti] = 0.f;
        for (i = 0; i < a_n; ++i) {
            float a1 = a[i];
            float a2 = a[i + 1];

            if ((t - a1) * (a2 - t) >= 0) {
                float y = a0_cache[i] + (t - a1) * (a0_cache[i+1] - a0_cache[i]) / (a2 - a1);
                if (y > ftp[ti]) ftp[ti] = y;
            }
        }
        ftp_buf[(k * n + attr_index) * ftp_buf_entry_sz + ti] = ftp[ti];
    }
}

__global__
void compute_b0_kernel(const float* __restrict__ fsets, const float* ftp_buf, const unsigned char* __restrict__ b_indices, float* b0_buf,
                       unsigned attr_index, unsigned b_len, unsigned b_n, unsigned N, unsigned n)
{
    unsigned i, ti, k;

    unsigned buf_entry_sz = WARP_MULTIPLE(MAX(T_NUM, b_n));

    extern __shared__ float cache[];

    float* ftp_cache = cache;
    float* b0 = ftp_cache + T_NUM;

    for (k = 0; k < N; ++k) {
        const float* __restrict__ b = fsets + b_indices[k] * b_n;

        b0[threadIdx.x] = 0.f;
        for (i = threadIdx.x; i < T_NUM; i += blockDim.x) ftp_cache[i] = ftp_buf[(k * n + attr_index) * buf_entry_sz + i];
#pragma unroll
        for (ti = 0; ti < T_NUM; ++ti) {
            float impl = IMPL((float)ti / (T_NUM - 1), b[threadIdx.x]);
            float min = MIN(ftp_cache[ti], impl);
            if (min > b0[threadIdx.x]) b0[threadIdx.x] = min;
        }
        b0_buf[(k * n + attr_index) * buf_entry_sz + threadIdx.x] = b0[threadIdx.x];
    }
}

__global__
void reduce_kernel(const float* b0_buf, float* partial_b0, unsigned b_n, unsigned N, unsigned n)
{
    unsigned buf_entry_sz = WARP_MULTIPLE(MAX(T_NUM, b_n));
    unsigned warp_multiple_dim = WARP_MULTIPLE(b_n); // same as (blockDim.x / n)
    unsigned i = threadIdx.x % warp_multiple_dim;
    unsigned attr_index = threadIdx.x / warp_multiple_dim;
    unsigned k = blockIdx.x;
    unsigned step;

    extern __shared__ float buf[];
    float* b0_buf_cache = buf;
    float* b0_cache = b0_buf_cache + blockDim.x;

    if (attr_index == 0) b0_cache[i] = 1.f;
    while (k < N) {
        b0_buf_cache[threadIdx.x] = b0_buf[(k * n + attr_index) * buf_entry_sz + i];
        for (step = 1; step < n; step <<= 1) {
            if ((attr_index & ((step<<1)-1)) == 0 && attr_index + step < n) {
                b0_buf_cache[i + attr_index * warp_multiple_dim]
                    = fmaxf(b0_buf_cache[i + attr_index * warp_multiple_dim],
                            b0_buf_cache[i + (attr_index + step) * warp_multiple_dim]);
            }
            __syncthreads();
        }
        // if (attr_index == 0) {
        //     float max = 0.f;
        //     for (unsigned j = 0; j < 6; ++j) max = fmaxf(max, b0_buf_cache[i + j * warp_multiple_dim]);
        //     // b0_cache[i] = fminf(b0_cache[i], b0_buf_cache[i]);
        //     b0_cache[i] = fminf(b0_cache[i], max);
        // }
        if (attr_index == 0) {
            b0_cache[i] = fminf(b0_cache[i], b0_buf_cache[i]);
        }
        // NOTE(sergey): So we skip synchronization in this point of loop,
        // because for debug purpose we thoughtlessly assume that 0 <= i < 32
        // and therefore will be executed by single warp.
        k += gridDim.x;
    }
    // NOTE(sergey): Make sure that b0_buf and partial_b0 don't overlap.
    // In other case memory layout compression can currupt, theoretically yet unused, data of other blocks.
    if (attr_index == 0) partial_b0[i + blockIdx.x * warp_multiple_dim] = b0_cache[i];
}

static cudaDeviceProp get_props_for_current_device()
{
    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    return prop;
}

#define LOG_DEVICE_PROP(p) \
    printf("prop." #p " == %d\n", prop.p)

extern "C"
void predict_gpu(const float** fsets[], const unsigned* fsets_lens, const unsigned* fsets_dims,
                 const float* a0_table[], const unsigned char* a_indices_table[],
                 const unsigned char* b_indices, float* b0, unsigned N, unsigned n)
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
     * | a0 table (Ptrs to buffer)         |
     * +-----------------------------------+
     * | a0 (data store)                   |
     * +-----------------------------------+
     * | a (indices in the F-sets table)   |
     * +-----------------------------------+
     * | b (indices in the F-sets table)   |
     * +-----------------------------------+
     * | Partial computs buffer (ftp | b0) |
     * | and final result  b0 (data store) |
     * +-----------------------------------+
     *
     * NOTE: N-th F-sets table entry corresponds to b's underling attribute.
     * NOTE: Everywhere below '*_table' varialbe name means array of pointers.
     */



    static cudaDeviceProp prop = get_props_for_current_device();

    unsigned i, j, k;
    unsigned partial_buf_entry_sz = WARP_MULTIPLE(MAX(T_NUM, fsets_dims[n]));
    unsigned fsets_buf_sz = 0;
    unsigned a0_buf_sz = 0;

    for (i = 0; i < n + 1; ++i) fsets_buf_sz += fsets_lens[i] * fsets_dims[i];
    for (i = 0; i < n; ++i) a0_buf_sz += fsets_dims[i];

    // NOTE(sergey): pl - page-locked.
    float *fsets_buf_pl, *fsets_buf_pl_table[n+1];
    float *a0_buf_pl, *a0_buf_pl_table[n];
    unsigned char *a_indices_pl, *b_indices_pl;
    unsigned fsets_buf_offset, a0_buf_offset;

    CU_HANDLE_ERROR(cudaHostAlloc(&fsets_buf_pl, sizeof(float[fsets_buf_sz]), cudaHostAllocPortable | cudaHostAllocWriteCombined));
    CU_HANDLE_ERROR(cudaHostAlloc(&a0_buf_pl, sizeof(float[a0_buf_sz]), cudaHostAllocPortable | cudaHostAllocWriteCombined));
    CU_HANDLE_ERROR(cudaHostAlloc(&a_indices_pl, sizeof(unsigned char[N * n]), cudaHostAllocPortable | cudaHostAllocWriteCombined));
    CU_HANDLE_ERROR(cudaHostAlloc(&b_indices_pl, sizeof(unsigned char[N]), cudaHostAllocPortable | cudaHostAllocWriteCombined));

    fsets_buf_offset = 0;
    for (i = 0; i < n + 1; ++i) {
        fsets_buf_pl_table[i] = fsets_buf_pl + fsets_buf_offset;
        for (j = 0; j < fsets_lens[i]; ++j) memcpy(fsets_buf_pl_table[i] + j * fsets_dims[i], fsets[i][j], sizeof(float[fsets_dims[i]]));
        fsets_buf_offset += fsets_lens[i] * fsets_dims[i];
    }
    a0_buf_offset = 0;
    for (i = 0; i < n; ++i) {
        a0_buf_pl_table[i] = a0_buf_pl + a0_buf_offset;
        memcpy(a0_buf_pl_table[i], a0_table[i], sizeof(float[fsets_dims[i]]));
        a0_buf_offset += fsets_dims[i];
    }
    for (i = 0; i < n; ++i) memcpy(a_indices_pl + i * N, a_indices_table[i], sizeof(unsigned char[N]));
    memcpy(b_indices_pl, b_indices, sizeof(unsigned char[N]));

    // NOTE(sergey): fsets_buf_table contains fsets_buf_d + offset values per attribute
    // and actually need only n entries (not n+1).
    float *fsets_buf_d, *fsets_buf_d_table[n];
    float *a0_buf_d, *a0_buf_d_table[n];
    unsigned char *a_indices_d, *b_indices_d;
    float *partial_buf_d;

    CU_HANDLE_ERROR(cudaMalloc(&fsets_buf_d, sizeof(float[fsets_buf_sz])));
    CU_HANDLE_ERROR(cudaMalloc(&a0_buf_d, sizeof(float[a0_buf_sz])));
    CU_HANDLE_ERROR(cudaMalloc(&a_indices_d, sizeof(unsigned char[N * n])));
    CU_HANDLE_ERROR(cudaMalloc(&b_indices_d, sizeof(unsigned char[N])));
    CU_HANDLE_ERROR(cudaMalloc(&partial_buf_d, sizeof(float[N * n][partial_buf_entry_sz])));

    cudaStream_t streams[n], helper_stream;

    CU_HANDLE_ERROR(cudaStreamCreate(&helper_stream));
    for (i = 0; i < n; ++i) CU_HANDLE_ERROR(cudaStreamCreate(streams + i));

    float* fsets_b_d = fsets_buf_d + fsets_buf_sz - fsets_lens[n] * fsets_dims[n];
    CU_HANDLE_ERROR(cudaMemcpyAsync(fsets_b_d, fsets_buf_pl_table[n], sizeof(float[fsets_lens[n] * fsets_dims[n]]),
                                    cudaMemcpyHostToDevice, helper_stream));
    CU_HANDLE_ERROR(cudaMemcpyAsync(b_indices_d, b_indices, sizeof(unsigned char[N]), cudaMemcpyHostToDevice, helper_stream));
    cudaStreamDestroy(helper_stream);

    fsets_buf_offset = 0;
    a0_buf_offset = 0;
    for (i = 0; i < n; ++i) {
        fsets_buf_d_table[i] = fsets_buf_d + fsets_buf_offset;
        CU_HANDLE_ERROR(cudaMemcpyAsync(fsets_buf_d_table[i], fsets_buf_pl_table[i], sizeof(float[fsets_lens[i] * fsets_dims[i]]), cudaMemcpyHostToDevice, streams[i]));
        a0_buf_d_table[i] = a0_buf_d + a0_buf_offset;
        CU_HANDLE_ERROR(cudaMemcpyAsync(a0_buf_d_table[i], a0_buf_pl_table[i], sizeof(float[fsets_dims[i]]), cudaMemcpyHostToDevice, streams[i]));
        CU_HANDLE_ERROR(cudaMemcpyAsync(a_indices_d + i * N, a_indices_pl + i * N, sizeof(unsigned char[N]), cudaMemcpyHostToDevice, streams[i]));

        fsets_buf_offset += fsets_lens[i] * fsets_dims[i];
        a0_buf_offset += fsets_dims[i];

        {
            unsigned blocks = prop.multiProcessorCount * 8;
            unsigned warp_multiple_dim = WARP_MULTIPLE(T_NUM);
            unsigned threads = warp_multiple_dim;
            unsigned shared_sz = sizeof(float[/* a_cache */ fsets_lens[i] * (fsets_dims[i]+1) + /* a0_cache */ (fsets_dims[i]+1) + /* ftp */ warp_multiple_dim]);
            compute_ftp_kernel<<<blocks, threads, shared_sz, streams[i]>>>(fsets_buf_d_table[i], a0_buf_d_table[i], a_indices_d + i * N, partial_buf_d,
                                                                           i, fsets_lens[i], fsets_dims[i], fsets_dims[n], N, n);
            CU_HANDLE_ERROR(cudaPeekAtLastError());
        }

        {
            unsigned blocks = prop.multiProcessorCount * 2;
            unsigned warp_multiple_dim = WARP_MULTIPLE(fsets_dims[n]);
            unsigned threads = warp_multiple_dim;
            unsigned shared_sz = sizeof(float[/* ftp_cache */ T_NUM + /* b0 */ warp_multiple_dim]);
            compute_b0_kernel<<<blocks, threads, shared_sz, streams[i]>>>(fsets_b_d, partial_buf_d, b_indices_d, partial_buf_d,
                                                                          i, fsets_lens[n], fsets_dims[n], N, n);
            CU_HANDLE_ERROR(cudaPeekAtLastError());
        }
    }

    for (i = 0; i < n; ++i) cudaStreamDestroy(streams[i]);
    cudaDeviceSynchronize();
    CU_HANDLE_ERROR(cudaPeekAtLastError());

    float* partial_b0_d;
    unsigned partial_b0_len, partial_b0_entry_sz;

    {
        float* b0_buf_d = partial_buf_d;

        // NOTE(sergey): Block count was taken from CUDA by Example book (Histogram computation with atomic operations),
        // where it was figured out experimentally that maximal performance is achieved,
        // when block number is exactly twice multiple of the number of multiprocessors.
        unsigned blocks = partial_b0_len = prop.multiProcessorCount * 8;
        unsigned warp_multiple_dim = partial_b0_entry_sz = WARP_MULTIPLE(fsets_dims[n]);
        unsigned threads = warp_multiple_dim * n;
        unsigned shared_sz = sizeof(float[n+1][warp_multiple_dim]);
        cudaMalloc(&partial_b0_d, sizeof(float[partial_b0_len][partial_b0_entry_sz]));
        reduce_kernel<<<blocks, threads, shared_sz>>>(b0_buf_d, partial_b0_d, fsets_dims[n], N, n);
        CU_HANDLE_ERROR(cudaPeekAtLastError());
    }

    float partial_b0[partial_b0_len][partial_b0_entry_sz];
    cudaMemcpy(partial_b0, partial_b0_d, sizeof(float[partial_b0_len][partial_b0_entry_sz]), cudaMemcpyDeviceToHost);
    // float partial_b0[N * n][partial_b0_entry_sz];
    // cudaMemcpy(partial_b0, partial_buf_d, sizeof(float[N * n][partial_buf_entry_sz]), cudaMemcpyDeviceToHost);

    memcpy(b0, partial_b0, sizeof(float[fsets_dims[n]]));
    for (i = 1; i < partial_b0_len; ++i) {
        for (j = 0; j < fsets_dims[n]; ++j) {
            // b0[j] = MIN(b0[j], partial_b0[i][j]);
            if (partial_b0[i][j] < b0[j]) b0[j] = partial_b0[i][j];
        }
    }

    // float b0_tmp[21];
    // for (j = 0; j < fsets_dims[n]; ++j) b0[j] = 1.f;
    // for (k = 0; k < N; ++k) {
    //     memset(b0_tmp, 0, sizeof(b0_tmp));
    //     for (i = 0; i < n; ++i) {
    //         for (j = 0; j < fsets_dims[n]; ++j) b0_tmp[j] = MAX(b0_tmp[j], partial_b0[i + k * n][j]);
    //     }
    //     for (j = 0; j < fsets_dims[n]; ++j) b0[j] = MIN(b0[j], b0_tmp[j]);
    // }

    cudaFree(fsets_buf_d);
    cudaFree(a0_buf_d);
    cudaFree(a_indices_d);
    cudaFree(b_indices_d);
    cudaFree(partial_buf_d);
    cudaFree(partial_b0_d);

    cudaFreeHost(fsets_buf_pl);
    cudaFreeHost(a0_buf_pl);
    cudaFreeHost(a_indices_pl);
    cudaFreeHost(b_indices_pl);
}
