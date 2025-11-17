# quirin/core/gf2.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np

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

    def copy(self) -> "BinaryMatrix":
        return self.clone()

    # low-level helpers
    def _xor_rows(self, src: int, tgt: int):
        # row_tgt ^= row_src
        self.rows[tgt] ^= self.rows[src]

    def apply_row_xor(self, src: int, tgt: int):
        if src == tgt:
            return
        self._xor_rows(src, tgt)

    def apply_cnot(self, control: int, target: int):
        if not (0 <= control < self.n and 0 <= target < self.n):
            raise IndexError("control/target out of range")
        self.apply_row_xor(control, target)

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

    def get_bit(self, row: int, col: int) -> int:
        if not (0 <= row < self.n and 0 <= col < self.n):
            raise IndexError("row/col out of range")
        word = col // 64
        bit = col % 64
        return (int(self.rows[row, word]) >> bit) & 1

    def count_off_diagonal_ones(self) -> int:
        diag = 0
        for i in range(self.n):
            diag += self.get_bit(i, i)
        return self.count_ones() - diag

    def is_identity(self) -> bool:
        for i in range(self.n):
            if not self.get_bit(i, i):
                return False
            if self.row_weight(i) != 1:
                return False
        return True

    def find_pivot_operations(self) -> List[Tuple[int, int]]:
        ops: List[Tuple[int, int]] = []
        for row in range(self.n):
            for word_idx in range(self.words):
                value = int(self.rows[row, word_idx])
                if not value:
                    continue
                base_col = word_idx * 64
                while value:
                    lsb = value & -value
                    bit_index = base_col + (lsb.bit_length() - 1)
                    if bit_index < self.n and bit_index != row:
                        ops.append((bit_index, row))
                    value ^= lsb
        return ops

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

    def evaluate_rowop_benefit(self, src: int, tgt: int) -> int:
        before = self.row_weight(tgt)
        after = 0
        for src_word, tgt_word in zip(self.rows[src], self.rows[tgt]):
            after += int(int(src_word) ^ int(tgt_word)).bit_count()
        return before - after

    def evaluate_operation_benefit(self, source: int, target: int) -> int:
        return self.evaluate_rowop_benefit(source, target)

    def apply_row_operation(self, src: int, tgt: int) -> Tuple[int, int]:
        self.apply_row_xor(src, tgt)
        return (src, tgt)

    def print_matrix(self):
        dense = self.to_dense_numpy()
        print("Matrix state:")
        for idx, row in enumerate(dense):
            row_bits = " ".join(str(bit) for bit in row)
            print(f"Row {idx}: {row_bits}")
        print(
            f"Ones count: {self.count_ones()}, Off-diagonal: {self.count_off_diagonal_ones()}"
        )
        print()

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
