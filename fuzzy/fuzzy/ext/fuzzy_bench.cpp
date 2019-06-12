
#include <stdio.h>
#include <benchmark/benchmark.h>

#include "fuzzy.h"
#include "fuzzy_bench_data.c"


#if 0
static void bench_predict_cpu_gcc(benchmark::State& state)
{
    float b0[fsets_dims[n]];

    for (auto _ : state) {
        unsigned N = state.range(0);
        predict_cpu(fsets_table, fsets_lens, fsets_dims, a0, a, b, b0, N, n);
    }
}
BENCHMARK(bench_predict_cpu_gcc)->Arg(N);

static void bench_predict_cpu_clang(benchmark::State& state)
{
    float b0[fsets_dims[n]];

    for (auto _ : state) {
        unsigned N = state.range(0);
        predict_cpu_asm_clang(fsets_table, fsets_lens, fsets_dims, a0, a, b, b0, N, n);
    }
}
BENCHMARK(bench_predict_cpu_clang)->Arg(N);

static void bench_predict_gpu(benchmark::State& state)
{
    float b0[n+1][fsets_dims[n]];

    for (auto _ : state) {
        unsigned N = state.range(0);
        predict_gpu(fsets_table, fsets_lens, fsets_dims, a0, a, b, (float*)b0, N, n);
    }
}
BENCHMARK(bench_predict_gpu)->Arg(N);

BENCHMARK_MAIN();

#else

int main(int argc, char* argv[])
{
    float b0[fsets_dims[n]];

    predict_gpu(fsets_table, fsets_lens, fsets_dims, a0, a, b, b0, N, n);
    for (unsigned i = 0; i < fsets_dims[n]; ++i) printf("%f ", (double)b0[i]);
    printf("\n");
    return b0[0];
}

#endif
