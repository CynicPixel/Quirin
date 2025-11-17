# quirin/core/gf2.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import math
import copy

# Bit-packed GF(2) matrix for CNOT linear transforms.
# Each row is stored as an array of np.uint64 words.
# This is optimized for speed on moderate n (n up to few thousands depending on memory).

def words_for_width(n_bits: int) -> int:
    return (n_bits + 63) // 64

class BinaryMatrix:
    def __init__(self, n: int):
        self.n = n
        self.words = words_for_width(n)
        # rows: shape (n, words), dtype=uint64
        self.rows = np.zeros((n, self.words), dtype=np.uint64)
        # initialize to identity matrix by setting bit i in row i
        for i in range(n):
            w = i // 64
            b = i % 64
            self.rows[i, w] = np.uint64(1 << b)

    @classmethod
    def from_cnot_sequence(cls, n: int, seq: List[Tuple[int, int]]):
        M = cls(n)
        for c, t in seq:
            M.apply_row_xor(c, t)
        return M

    def clone(self) -> "BinaryMatrix":
        M = BinaryMatrix(self.n)
        M.rows = self.rows.copy()
        return M

    # low-level helpers
    def _xor_rows(self, src: int, tgt: int):
        # row_tgt ^= row_src
        self.rows[tgt] ^= self.rows[src]

    def apply_row_xor(self, src: int, tgt: int):
        if src == tgt:
            return
        self._xor_rows(src, tgt)

    def apply_rowops_sequence(self, seq: List[Tuple[int, int]]):
        for src, tgt in seq:
            self.apply_row_xor(src, tgt)

    def count_ones(self) -> int:
        # vectorized popcount across all words
        # convert rows to uint8 view then use np.unpackbits for popcount per byte.
        # For moderate sizes this is fast and avoids Python loop popcount.
        arr = self.rows.view(np.uint8)
        # arr shape = (n, words*8)
        return int(np.unpackbits(arr).sum())

    def row_weight(self, r: int) -> int:
        # popcount of row r
        row_words = self.rows[r]
        # convert each uint64 to Python int and use bit_count (fast in CPython)
        # loop over words (small number) - preferable to unpackbits for single row
        cnt = 0
        for w in row_words:
            cnt += int(w).bit_count()
        return cnt

    def to_dense_numpy(self) -> np.ndarray:
        """Return an n x n uint8 numpy array (useful for small n debugging)."""
        out = np.zeros((self.n, self.n), dtype=np.uint8)
        for i in range(self.n):
            for w in range(self.words):
                v = int(self.rows[i, w])
                if v:
                    base = w * 64
                    while v:
                        lsb = (v & -v).bit_length() - 1
                        idx = base + lsb
                        if idx < self.n:
                            out[i, idx] = 1
                        v &= v - 1
        return out

    def equals(self, other: "BinaryMatrix") -> bool:
        if self.n != other.n:
            return False
        return bool((self.rows ^ other.rows).sum() == 0)

    def to_cnot_sequence(self, rowops: List[Tuple[int, int]]) -> List[Tuple[int,int]]:
        # rowops represent R_t <- R_t + R_s ; convert directly into CNOT (control=src,target=tgt)
        # caller should ensure rowops represent the sequence in correct order (I -> T or T -> I)
        return [(s, t) for (s, t) in rowops]

    def compute_transform_of_basis(self) -> List[int]:
        """
        Return list where i-th entry is integer (bitmask) representing where basis vector e_i maps to.
        Useful debug helper: for each column j, what is image?
        Note: returns Python ints representing bits across n bits.
        """
        masks = []
        for i in range(self.n):
            # compute column i by reading bit i from each row
            col_mask = 0
            for r in range(self.n):
                w = i // 64
                b = i % 64
                if (int(self.rows[r, w]) >> b) & 1:
                    col_mask |= (1 << r)
            masks.append(col_mask)
        return masks

    # Utility for applying a cnot sequence to this matrix starting from identity:
    @staticmethod
    def matrix_from_cnot_seq(n: int, seq: List[Tuple[int,int]]) -> "BinaryMatrix":
        return BinaryMatrix.from_cnot_sequence(n, seq)

    # Pretty debug
    def __repr__(self):
        return f"<BinaryMatrix n={self.n} words={self.words}>"
