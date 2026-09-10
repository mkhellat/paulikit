"""Pauli decomposition algorithm implementations.

``fwht`` holds the decomposition itself, based on the Fast
Walsh-Hadamard Transform, O(N^2 log N) for an N x N matrix.
``autotune`` sizes chunks against the machine's measured cache
hierarchy and available memory.
"""
