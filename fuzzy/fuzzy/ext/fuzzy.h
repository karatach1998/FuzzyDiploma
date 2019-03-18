#ifndef FUZZY_H
#define FUZZY_H

#define T_NUM 11

#ifdef __cplusplus
extern "C"
#endif
void predict_gpu(const float** fsets_table[], const unsigned* fsets_lens, const unsigned* fsets_dims,
                 const float* a0[], const unsigned char* a[], const unsigned char* b, float* b0,
                 unsigned N, unsigned n);

#endif // FUZZY_H