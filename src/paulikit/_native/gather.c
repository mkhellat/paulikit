/* Original C implementation of the fully-dense per-chunk XOR-gather.
 * See gather.h for the design rationale. */

#include "gather.h"

void paulikit_gather_dense_chunk(
    const double *operator_data,
    int64_t dim,
    int64_t chunk_start,
    int64_t rows,
    double *out_data
) {
    if (dim < 1 || rows < 1) {
        return;
    }

    /* Complex128 = two adjacent doubles per element, so every row is
     * dim*2 doubles wide and every element is 2 doubles - working in
     * doubles throughout (as wht.c's stage loop does) avoids any
     * dependence on how `double complex` is passed and gives the
     * compiler a longer, simpler contiguous copy to vectorize. */
    const int64_t dim_doubles = dim * 2;

    for (int64_t row = 0; row < rows; row++) {
        const int64_t x = chunk_start + row;
        double *restrict out_row = out_data + row * dim_doubles;

        /* p = x ^ q, for q = 0..dim-1: XOR by the constant x is a
         * bijection over [0, dim), so this visits exactly the dim
         * cells operator[x^q][q] - one read per output cell, no
         * skipped or repeated source cell. */
        for (int64_t q = 0; q < dim; q++) {
            const int64_t p = x ^ q;
            const double *restrict src = operator_data + (p * dim + q) * 2;
            out_row[q * 2] = src[0];
            out_row[q * 2 + 1] = src[1];
        }
    }
}
