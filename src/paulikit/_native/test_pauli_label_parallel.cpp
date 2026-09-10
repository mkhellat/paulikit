/* Standalone C++ correctness + timing check for
 * pauli_label_batch_parallel, run before any Python binding work.
 * Compares against the serial pauli_label_batch term-by-term (not
 * just spot checks) across several n_terms/n_qubits combinations,
 * then reports a rough serial-vs-parallel speedup on a large batch.
 * Build/run:
 *
 *   c++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test_pauli_label_parallel \
 *       pauli_label.c pauli_label_parallel.cpp \
 *       test_pauli_label_parallel.cpp -ltbb \
 *       && /tmp/test_pauli_label_parallel
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "pauli_label.h"
#include "pauli_label_parallel.h"

static int failures = 0;

static void check_matches_serial(int64_t n_terms, int n_qubits, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, (1u << n_qubits) - 1);

    std::vector<uint32_t> xs(n_terms), zs(n_terms);
    for (int64_t i = 0; i < n_terms; i++) {
        xs[i] = dist(rng);
        zs[i] = dist(rng);
    }

    std::vector<char> serial_out(n_terms * n_qubits);
    std::vector<char> parallel_out(n_terms * n_qubits);

    pauli_label_batch(xs.data(), zs.data(), n_terms, n_qubits, serial_out.data());
    pauli_label_batch_parallel(xs.data(), zs.data(), n_terms, n_qubits, parallel_out.data());

    if (std::memcmp(serial_out.data(), parallel_out.data(), n_terms * n_qubits) != 0) {
        printf(
            "FAIL: parallel batch diverges from serial batch "
            "(n_terms=%lld, n_qubits=%d, seed=%u)\n",
            (long long)n_terms, n_qubits, seed
        );
        failures++;
    }
}

int main(void) {
    /* Small and mid-sized batches, several n_qubits, several seeds -
     * correctness must hold regardless of how work is chunked across
     * threads. */
    for (unsigned seed = 0; seed < 5; seed++) {
        check_matches_serial(1, 3, seed);
        check_matches_serial(7, 5, seed);
        check_matches_serial(1000, 8, seed);
        check_matches_serial(50000, 11, seed);
    }

    if (failures == 0) {
        printf("All pauli_label_batch_parallel correctness checks passed.\n");
    } else {
        printf("%d correctness check(s) failed.\n", failures);
    }

    /* Rough serial-vs-parallel timing at a size representative of the
     * N=50 matched-benchmark case (real numbers with the actual
     * Hamiltonian belong in the Python-level benchmark, not here -
     * this is just a sanity check that parallelization is doing
     * something on this machine). */
    const int64_t n_terms = 1'261'568;
    const int n_qubits = 11;
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, (1u << n_qubits) - 1);
    std::vector<uint32_t> xs(n_terms), zs(n_terms);
    for (int64_t i = 0; i < n_terms; i++) {
        xs[i] = dist(rng);
        zs[i] = dist(rng);
    }
    std::vector<char> out(n_terms * n_qubits);

    auto t0 = std::chrono::steady_clock::now();
    pauli_label_batch(xs.data(), zs.data(), n_terms, n_qubits, out.data());
    auto t1 = std::chrono::steady_clock::now();
    pauli_label_batch_parallel(xs.data(), zs.data(), n_terms, n_qubits, out.data());
    auto t2 = std::chrono::steady_clock::now();

    double serial_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double parallel_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf(
        "n_terms=%lld n_qubits=%d serial=%.2fms parallel=%.2fms speedup=%.2fx\n",
        (long long)n_terms, n_qubits, serial_ms, parallel_ms, serial_ms / parallel_ms
    );

    return failures == 0 ? 0 : 1;
}
