#ifndef FUZZY_H
#define FUZZY_H

#define T_DIM 11

#ifdef __cplusplus
extern "C" {
#endif

void predict_cpu(const float** fsets_table[], const unsigned* fsets_lens, const unsigned* fsets_dims,
                 const float* a0[], const unsigned char* a[], const unsigned char* b, float* b0,
                 unsigned N, unsigned n);

extern
void predict_cpu_asm_clang(const float** fsets_table[], const unsigned* fsets_lens, const unsigned* fsets_dims,
                           const float* a0[], const unsigned char* a[], const unsigned char* b, float* b0,
                           unsigned N, unsigned n);

void predict_gpu(const float** fsets_table[], const unsigned* fsets_lens, const unsigned* fsets_dims,
                 const float* a0[], const unsigned char* a[], const unsigned char* b, float* b0,
                 unsigned N, unsigned n);

#ifdef __cplusplus
}
#endif

#endif // FUZZY_H
