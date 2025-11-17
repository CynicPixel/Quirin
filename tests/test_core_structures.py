"""Core data structure regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quirin.core.ast import CircuitAST, Gate
from quirin.core import gf2
from quirin.io import qiskit_io, stim_io


def _gate_signature(ast: CircuitAST):
    return [(gate.kind, gate.qubits) for gate in ast.gates]


def test_circuit_ast_copy_is_independent():
    ast = CircuitAST(2)
    ast.append(Gate("H", (0,), {"label": "h0"}))
    cloned = ast.copy()

    assert ast is not cloned
    assert ast.gates is not cloned.gates
    assert _gate_signature(ast) == _gate_signature(cloned)

    cloned.gates[0].params["label"] = "mutated"
    assert ast.gates[0].params["label"] == "h0"


def test_circuit_ast_cnot_blocks_detection():
    ast = CircuitAST(3)
    ast.extend(
        [
            Gate("CNOT", (0, 1)),
            Gate("CNOT", (1, 2)),
            Gate("H", (0,)),
            Gate("CNOT", (2, 1)),
        ]
    )

    blocks = ast.to_cnot_blocks()
    assert blocks == [(0, 2), (3, 4)]


def test_binary_matrix_from_cnot_seq_matches_expected():
    seq = [(0, 1), (1, 2)]
    matrix = gf2.BinaryMatrix.matrix_from_cnot_seq(3, seq)
    dense = matrix.to_dense_numpy()

    expected = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(dense, expected)


@pytest.mark.skipif(not stim_io._STIM_AVAILABLE, reason="stim not installed")
def test_ast_to_stim_round_trip_preserves_gate_sequence():
    ast = CircuitAST(2)
    ast.extend([Gate("H", (0,)), Gate("CNOT", (0, 1)), Gate("S", (1,))])

    circuit = stim_io.circuit_to_stim(ast)
    rebuilt = stim_io.circuit_from_stim(circuit)

    assert _gate_signature(rebuilt) == _gate_signature(ast)


@pytest.mark.skipif(not qiskit_io.is_qiskit_available(), reason="qiskit not installed")
def test_ast_to_qiskit_round_trip_preserves_gate_sequence():
    ast = CircuitAST(2)
    ast.extend([Gate("H", (0,)), Gate("CNOT", (0, 1)), Gate("Z", (1,))])

    qc = qiskit_io.circuit_to_qiskit(ast)
    rebuilt = qiskit_io.circuit_from_qiskit(qc)

    assert _gate_signature(rebuilt) == _gate_signature(ast)