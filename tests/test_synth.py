from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quirin.core.gf2 import BinaryMatrix
from quirin.synth.paper_greedy import (
    PaperGreedyOptimizer,
    PaperGreedySynth,
    _reduce_to_identity,
)


def _apply_ops_to_identity(n_qubits: int, ops: Iterable[Tuple[int, int]]) -> BinaryMatrix:
    matrix = BinaryMatrix(n_qubits)
    matrix.apply_rowops_sequence(list(ops))
    return matrix


def _build_target_matrix(n_qubits: int, gates: Iterable[Tuple[int, int]]) -> BinaryMatrix:
    return BinaryMatrix.matrix_from_cnot_seq(n_qubits, list(gates))


def test_reduce_to_identity_simple_chain():
    n_qubits = 3
    gates = [(0, 1), (1, 2), (0, 2)]
    target_matrix = _build_target_matrix(n_qubits, gates)

    working_matrix = target_matrix.clone()
    row_ops = _reduce_to_identity(working_matrix)

    assert working_matrix.is_identity()
    assert row_ops, "expected at least one reduction step"

    rebuilt = _apply_ops_to_identity(n_qubits, reversed(row_ops))
    assert rebuilt.equals(target_matrix)


def test_paper_greedy_optimizer_roundtrip():
    n_qubits = 4
    gates = [(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)]

    optimizer = PaperGreedyOptimizer()
    optimized, stats = optimizer.optimize(n_qubits, gates)

    assert stats["original_gates"] == len(gates)
    assert stats["optimized_gates"] == len(optimized)

    original_matrix = _build_target_matrix(n_qubits, gates)
    optimized_matrix = _build_target_matrix(n_qubits, optimized)
    assert original_matrix.equals(optimized_matrix)


def test_paper_greedy_synth_matches_optimizer_sequences():
    circuits: List[Tuple[int, List[Tuple[int, int]]]] = [
        (2, [(0, 1)]),
        (3, [(0, 1), (1, 2)]),
        (3, [(0, 1), (0, 2), (1, 2)]),
        (4, [(0, 1), (2, 3), (1, 3), (0, 2)]),
    ]

    optimizer = PaperGreedyOptimizer()
    synth = PaperGreedySynth()

    for n_qubits, gates in circuits:
        optimized, _ = optimizer.optimize(n_qubits, gates)
        target_matrix = _build_target_matrix(n_qubits, gates)

        synthesized = synth.synthesize(target_matrix)
        assert synthesized == optimized

        rebuilt = _apply_ops_to_identity(n_qubits, synthesized)
        assert rebuilt.equals(target_matrix)
