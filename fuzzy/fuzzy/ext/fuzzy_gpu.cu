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
void compute_kernel(const float* __restrict__ a_fsets, const float* __restrict__ b_fsets, const float* __restrict__ a0,
                    const unsigned char* __restrict__ a_indices, const unsigned char* __restrict__ b_indices, float* b0_buf,
                    unsigned attr_index, unsigned a_len, unsigned a_n, unsigned b_len, unsigned b_n, unsigned N, unsigned n)
{
    unsigned i, j, k;
    unsigned buf_entry_sz = WARP_MULTIPLE(b_n);
    unsigned warp_multiple_dim = WARP_MULTIPLE(MAX(T_NUM, b_n));
    unsigned executor_per_block = blockDim.x / warp_multiple_dim;
    unsigned executor_index = threadIdx.x / warp_multiple_dim;
    unsigned tid = threadIdx.x % warp_multiple_dim;

    extern __shared__ float cache[];

    float* ftp = cache + executor_index * (WARP_MULTIPLE(T_NUM) + WARP_MULTIPLE(b_n));
    float* b0 = cache + executor_index * (WARP_MULTIPLE(T_NUM) + WARP_MULTIPLE(b_n)) + WARP_MULTIPLE(T_NUM);
    
    unsigned ti = tid;
    const float t = (float) ti / (T_NUM - 1);

    for (k = blockIdx.x * executor_per_block + executor_index; k < N; k += gridDim.x * executor_per_block) {

        // NOTE(sergey): both ftp and b0 shared memory buffer's size is multiple of the warp size.
        // Underlying thought about warps organizaton in block of this kernel is
        // that some warps will be take off from execution by SM's warp scheduler
        // due to if condition. It guards from overhead when computing with unified block size
        // (maximum from ftp and b0 buffer sizes).

        if (tid < WARP_MULTIPLE(T_NUM)) {
            const float* __restrict__ a = a_fsets + a_indices[k] * a_n;
            const unsigned ti = tid;

            ftp[ti] = 0.f;
            for (i = 0; i < a_n - 1; ++i) {
                float a1 = a[i];
                float a2 = a[i + 1];

                // NOTE(sergey): Experimentally was approved that performing arithmetic
                // before condition check is more efficient that doing it in the true case.
                // I guess it happens because condition check followed by assignment translates
                // to appropriate PTX condition-move instruction.
                float y = a0[i] + (t - a1) * (a0[i+1] - a0[i]) / (a2 - a1);
                if ((t - a1) * (a2 - t) >= 0 && y > ftp[ti]) ftp[ti] = y;
            }
        }

        if (tid < WARP_MULTIPLE(b_n)) {
            const float* __restrict__ b = b_fsets + b_indices[k] * b_n;

            b0[tid] = 0.f;
#pragma unroll
            for (ti = 0; ti < T_NUM; ++ti) {
                float impl = IMPL((float)ti / (T_NUM - 1), b[tid]);
                float min = fminf(ftp[ti], impl);
                if (min > b0[tid]) b0[tid] = min;
            }
            b0_buf[(k * n + attr_index) * buf_entry_sz + tid] = b0[tid];
        }
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
     * | Fuzzy sets buffer (data store)    |
     * +-----------------------------------+
     * | a0 buffer (data store)            |
     * +-----------------------------------+
     * | a (F-sets attr table indices)     |
     * +-----------------------------------+
     * | b (F-sets attr table indices)     |
     * +-----------------------------------+
     * | Partial b0 computations buffer    |
     * +-----------------------------------+
     *
     * NOTE: n-th F-sets table entry corresponds to b's underling attribute.
     * NOTE: Everywhere below '*_table' varialbe name means array of pointers.
     */



    static cudaDeviceProp prop = get_props_for_current_device();

    unsigned i, j;
    unsigned partial_buf_entry_sz = WARP_MULTIPLE(fsets_dims[n]);
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
    // TODO(sergey): Implement waiting for the copy finished event to synchronize each of streams.
    cudaDeviceSynchronize();

    fsets_buf_offset = 0;
    a0_buf_offset = 0;
    for (i = 0; i < n; ++i) {
        {
            fsets_buf_d_table[i] = fsets_buf_d + fsets_buf_offset;
            CU_HANDLE_ERROR(cudaMemcpyAsync(fsets_buf_d_table[i], fsets_buf_pl_table[i], sizeof(float[fsets_lens[i] * fsets_dims[i]]), cudaMemcpyHostToDevice, streams[i]));
            a0_buf_d_table[i] = a0_buf_d + a0_buf_offset;
            CU_HANDLE_ERROR(cudaMemcpyAsync(a0_buf_d_table[i], a0_buf_pl_table[i], sizeof(float[fsets_dims[i]]), cudaMemcpyHostToDevice, streams[i]));
            CU_HANDLE_ERROR(cudaMemcpyAsync(a_indices_d + i * N, a_indices_pl + i * N, sizeof(unsigned char[N]), cudaMemcpyHostToDevice, streams[i]));

            fsets_buf_offset += fsets_lens[i] * fsets_dims[i];
            a0_buf_offset += fsets_dims[i];
        }

        {
            unsigned min_blocks_per_sm = prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock;
            unsigned blocks = prop.multiProcessorCount * min_blocks_per_sm;
            // NOTE(sergey): To achieve better performance we probably need to perform computation for more than one rule inside one block.
            // So we use several executors per block.
            unsigned warp_multiple_dim = WARP_MULTIPLE(MAX(T_NUM, fsets_dims[n]));
            unsigned executor_per_block = prop.maxThreadsPerBlock / warp_multiple_dim;
            unsigned threads = executor_per_block * warp_multiple_dim;
            unsigned shared_sz = sizeof(float[executor_per_block][/* ftp */ WARP_MULTIPLE(T_NUM) + /* b0 */ WARP_MULTIPLE(fsets_dims[n])]);
            compute_kernel<<<blocks, threads, shared_sz, streams[i]>>>(fsets_buf_d_table[i], fsets_b_d, a0_buf_d_table[i], a_indices_d + i * N, b_indices_d, partial_buf_d,
                                                                       i, fsets_lens[i], fsets_dims[i], fsets_lens[n], fsets_dims[n], N, n);
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

    memcpy(b0, partial_b0, sizeof(float[fsets_dims[n]]));
    for (i = 1; i < partial_b0_len; ++i) {
        for (j = 0; j < fsets_dims[n]; ++j) {
            if (partial_b0[i][j] < b0[j]) b0[j] = partial_b0[i][j];
        }
    }

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
