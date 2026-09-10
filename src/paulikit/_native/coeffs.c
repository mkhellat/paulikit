/* Fused phase/scale/threshold/emit kernel. See coeffs.h for the
 * design rationale and the profile that motivated it. */

#include "coeffs.h"

/* popcount: use the builtin where the compiler offers one (GCC and
 * Clang both do, and both lower it to a POPCNT instruction when the
 * target supports it), otherwise a portable SWAR fallback. Only the
 * low two bits of the result are ever used, but as Hacker's Delight
 * section 5-1 makes clear, a carry-save reduction cannot be truncated
 * to produce them more cheaply - carries out of the lower stages
 * propagate into bits 0-1 - so there is no shortcut to take here. */
#if defined(__GNUC__) || defined(__clang__)
#define PAULIKIT_POPCOUNT(v) ((unsigned)__builtin_popcountll((unsigned long long)(v)))
#else
static inline unsigned paulikit_popcount_fallback(uint64_t v)
{
    v = v - ((v >> 1) & 0x5555555555555555ULL);
    v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    v = (v + (v >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return (unsigned)((v * 0x0101010101010101ULL) >> 56);
}
#define PAULIKIT_POPCOUNT(v) paulikit_popcount_fallback((uint64_t)(v))
#endif

/* The body is a macro rather than a function taking a width argument
 * so the index width is a compile-time constant in each instantiation:
 * the store becomes a single typed move with no branch in the inner
 * loop. Three instantiations cost a few hundred bytes of text and keep
 * the hot loop free of per-element dispatch. */
#define PAULIKIT_EMIT_LOOP(IDX_T)                                             \
    do {                                                                      \
        IDX_T *restrict xs_out = (IDX_T *)out_x;                              \
        IDX_T *restrict zs_out = (IDX_T *)out_z;                              \
        for (int64_t r = 0; r < rows; r++) {                                  \
            const int64_t xr = row_x[r];                                      \
            const double *restrict row = data + r * dim * 2;                  \
            const IDX_T x_emit = (IDX_T)xr;                                   \
            for (int64_t z = 0; z < dim; z++) {                               \
                const double re = row[2 * z];                                 \
                const double im = row[2 * z + 1];                             \
                                                                              \
                /* conj(i**popcount(x & z)) applied as a swap/negate  */      \
                /* pair - no multiply, no table, no memory traffic.   */      \
                double cr, ci;                                                \
                switch (PAULIKIT_POPCOUNT(xr & z) & 3u) {                     \
                    case 0:  cr =  re; ci =  im; break;                       \
                    case 1:  cr =  im; ci = -re; break;                       \
                    case 2:  cr = -re; ci = -im; break;                       \
                    default: cr = -im; ci =  re; break;                       \
                }                                                             \
                cr *= inv_dim;                                                \
                ci *= inv_dim;                                                \
                                                                              \
                /* |c| > atol  <=>  |c|^2 > atol^2 for atol >= 0, so   */     \
                /* the square root is avoidable and exactly so.        */     \
                if (cr * cr + ci * ci > atol_squared) {                       \
                    xs_out[n] = x_emit;                                       \
                    zs_out[n] = (IDX_T)z;                                     \
                    out_coeff[2 * n] = cr;                                    \
                    out_coeff[2 * n + 1] = ci;                                \
                    n++;                                                      \
                }                                                             \
            }                                                                 \
        }                                                                     \
    } while (0)

int64_t paulikit_coeffs_from_transformed(
    const double *data,
    int64_t rows,
    int64_t dim,
    const int64_t *row_x,
    double inv_dim,
    double atol,
    int index_bytes,
    void *out_x,
    void *out_z,
    double *out_coeff
) {
    int64_t n = 0;

    if (rows < 1 || dim < 1) {
        return 0;
    }

    const double atol_squared = atol * atol;

    switch (index_bytes) {
        case 2:
            PAULIKIT_EMIT_LOOP(uint16_t);
            break;
        case 4:
            PAULIKIT_EMIT_LOOP(uint32_t);
            break;
        default:
            PAULIKIT_EMIT_LOOP(int64_t);
            break;
    }

    return n;
}

#undef PAULIKIT_EMIT_LOOP
