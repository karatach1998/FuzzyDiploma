#include <memory.h>
#include "fuzzy.h"

#define CACHE_LINE  32
#ifdef _MSC_VER
#define CACHE_ALIGN __declspec(align(CACHE_LINE))
#else
#define CACHE_ALIGN __attribute__((aligned(CACHE_LINE)))
#endif

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define IMPL(a, b) (1 - (a) + (b))


// NOTE(sergey): Make sure, when calling this subroutine, all input arrays have 64-byte alignment.
void predict_cpu(const float** fsets_table[], const unsigned* fsets_lens, const unsigned* fsets_dims,
                 const float* a0[], const unsigned char* a_indices[], const unsigned char* b_indices, float* b0,
                 unsigned N, unsigned n)
{
    // NOTE(sergey): In this function we assume following:
    // - all real fuzzy sets data, pointed by fsets_table, are cache-line aligned;
    // - both a, b are cache-line aligned.

    unsigned i, j, k, ti;
    float ftp[T_NUM], b0_row[fsets_dims[n]];

    for (j = 0; j < fsets_dims[n]; ++j) b0[j] = 1.f;
    for (k = 0; k < N; ++k) {
        // Compute attribute union [max for each i=[1,n)]
        memset(b0_row, 0, sizeof(b0_row));
        for (i = 0; i < n; ++i) {
            const float* a = fsets_table[i][a_indices[i][k]];
            const float* b = fsets_table[n][b_indices[k]];
            const float* a0i = a0[i];

            // Compute fuzzy truth power
            memset(ftp, 0, sizeof(ftp));
            for (ti = 0; ti < T_NUM; ++ti) {
                float t = (float) ti / (T_NUM - 1);

                for (j = 0; j < fsets_dims[i] - 1; ++j) {
                    float a1 = a[j];
                    float a2 = a[j + 1];

                    if ((t - a1) * (a2 - t) >= 0) {
                        // float x = x1 + (t - a1) * (x2 - x1) / (a2 - a1);
                        // float y = a0[j] + (x - x1) * (a0[j + 1] - a0[j]) / (x2 - x1);
                        float y = a0i[j] + (t - a1) * (a0i[j + 1] - a0i[j]) / (a2 - a1);
                        if (y > ftp[ti]) ftp[ti] = y;
                    }
                }
            }

            // Compute implication
            // Compute t-norm reduction
            for (j = 0; j < fsets_dims[n]; ++j) {
                float max = 0.f;
                for (ti = 0; ti < T_NUM; ++ti) max = MAX(max, MIN(ftp[ti], IMPL((float) ti / (T_NUM - 1), b[j])));
                b0_row[j] = MAX(b0_row[j], max);
            }
        }

        for (j = 0; j < fsets_dims[n]; ++j) b0[j] = MIN(b0[j], b0_row[j]);
    }
}
