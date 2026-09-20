/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com> */

/* Original C implementation of the Hermiticity-violation check.
 * See hermitian_check.h for the design rationale. */

#include "hermitian_check.h"

int64_t paulikit_first_hermitian_violation(
    const double *data,
    int64_t n,
    double atol
) {
    if (n < 1) {
        return -1;
    }

    const double atol_sq = atol * atol;

    for (int64_t i = 0; i < n; i++) {
        const double re = data[2 * i];
        const double im = data[2 * i + 1];
        const double im_sq = im * im;
        const double mag_sq = re * re + im_sq;
        const double thresh_sq = atol_sq > 1e-12 * mag_sq
            ? atol_sq
            : 1e-12 * mag_sq;
        if (im_sq > thresh_sq) {
            return i;
        }
    }
    return -1;
}
