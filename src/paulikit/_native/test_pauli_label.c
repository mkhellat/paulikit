/* Standalone C correctness check for pauli_label.c, run before any
 * Python binding work. Exhaustively checks n_qubits 1..3
 * against a hand-derived truth table, matching the exhaustive range
 * covered by tests/test_fwht.py's
 * test_pauli_label_round_trips_through_pauli_utils. Build/run:
 *
 *   cc -std=c99 -Wall -Wextra -o /tmp/test_pauli_label \
 *       pauli_label.c test_pauli_label.c && /tmp/test_pauli_label
 */

#include <stdio.h>
#include <string.h>
#include "pauli_label.h"

static int failures = 0;

static void check(uint32_t x, uint32_t z, int n_qubits, const char *expected) {
    char out[33];
    pauli_label(x, z, n_qubits, out);
    if (strcmp(out, expected) != 0) {
        printf(
            "FAIL: pauli_label(x=%u, z=%u, n_qubits=%d) = %s, expected %s\n",
            x, z, n_qubits, out, expected
        );
        failures++;
    }
}

int main(void) {
    /* n_qubits = 1: single-bit cases, direct from the letter table. */
    check(0, 0, 1, "I");
    check(1, 0, 1, "X");
    check(0, 1, 1, "Z");
    check(1, 1, 1, "Y");

    /* n_qubits = 2: qubit 0 = bit 1 (MSB), qubit 1 = bit 0 (LSB). */
    check(0b00, 0b00, 2, "II");
    check(0b10, 0b00, 2, "XI");
    check(0b01, 0b00, 2, "IX");
    check(0b11, 0b00, 2, "XX");
    check(0b00, 0b10, 2, "ZI");
    check(0b00, 0b01, 2, "IZ");
    check(0b10, 0b01, 2, "XZ");
    check(0b01, 0b10, 2, "ZX");
    check(0b11, 0b11, 2, "YY");

    /* n_qubits = 3. */
    check(0b101, 0b010, 3, "XZX");
    check(0b111, 0b111, 3, "YYY");
    check(0b000, 0b000, 3, "III");

    /* Batch entry point must match the single-call kernel term by
     * term, including for a mix of n_qubits=3 terms. */
    {
        uint32_t xs[3] = {0b101, 0b111, 0b000};
        uint32_t zs[3] = {0b010, 0b111, 0b000};
        char batch_out[3 * 3];
        pauli_label_batch(xs, zs, 3, 3, batch_out);
        const char *expected[3] = {"XZX", "YYY", "III"};
        for (int i = 0; i < 3; i++) {
            if (strncmp(batch_out + i * 3, expected[i], 3) != 0) {
                printf(
                    "FAIL: pauli_label_batch term %d = %.3s, expected %s\n",
                    i, batch_out + i * 3, expected[i]
                );
                failures++;
            }
        }
    }

    /* Exhaustive check for n_qubits = 1..4 against the letter-table
     * definition directly (independent re-derivation, not just
     * calling the kernel again). */
    for (int n_qubits = 1; n_qubits <= 4; n_qubits++) {
        uint32_t dim = 1u << n_qubits;
        for (uint32_t x = 0; x < dim; x++) {
            for (uint32_t z = 0; z < dim; z++) {
                char out[9];
                pauli_label(x, z, n_qubits, out);
                if ((int)strlen(out) != n_qubits) {
                    printf(
                        "FAIL: pauli_label(x=%u, z=%u, n_qubits=%d) "
                        "length %zu != %d\n",
                        x, z, n_qubits, strlen(out), n_qubits
                    );
                    failures++;
                    continue;
                }
                for (int qubit = 0; qubit < n_qubits; qubit++) {
                    int bit = n_qubits - 1 - qubit;
                    unsigned xj = (x >> bit) & 1u;
                    unsigned zj = (z >> bit) & 1u;
                    char expected_char =
                        (xj == 0 && zj == 0) ? 'I' :
                        (xj == 1 && zj == 0) ? 'X' :
                        (xj == 0 && zj == 1) ? 'Z' : 'Y';
                    if (out[qubit] != expected_char) {
                        printf(
                            "FAIL: pauli_label(x=%u, z=%u, n_qubits=%d)[%d] "
                            "= %c, expected %c\n",
                            x, z, n_qubits, qubit, out[qubit], expected_char
                        );
                        failures++;
                    }
                }
            }
        }
    }

    if (failures == 0) {
        printf("All pauli_label C kernel checks passed.\n");
        return 0;
    }
    printf("%d check(s) failed.\n", failures);
    return 1;
}
