"""Unit tests for :mod:`quirin.verifier`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quirin.core.ast import CircuitAST, Gate
from quirin import verifier
from quirin.io import qiskit_io


def make_ast(n_qubits: int, ops):
    ast = CircuitAST(n_qubits)
    for kind, qubits in ops:
        ast.append(Gate(kind, tuple(qubits)))
    return ast


@pytest.mark.skipif(not verifier._STIM_AVAILABLE, reason="stim not installed")
def test_stim_tableau_equivalence_detects_match_and_difference():
    ast_a = make_ast(2, [("H", (0,)), ("CNOT", (0, 1))])
    ast_b = make_ast(2, [("H", (0,)), ("CNOT", (0, 1))])
    assert verifier.is_clifford_only(ast_a)

    equal, message = verifier.equivalent(ast_a, ast_b)
    assert equal is True
    assert "stim" in message.lower()

    ast_c = make_ast(2, [("H", (0,)), ("S", (0,)), ("CNOT", (0, 1))])
    different, diff_message = verifier.equivalent(ast_a, ast_c)
    assert different is False
    assert "stim" in diff_message.lower()


def test_cnot_matrix_equivalence_checks_binary_matrix():
    ast_a = make_ast(3, [("CNOT", (0, 1)), ("CNOT", (1, 2))])
    ast_b = make_ast(3, [("CNOT", (0, 1)), ("CNOT", (1, 2))])

    equal, message = verifier.equivalent(ast_a, ast_b)
    assert equal is True
    assert "gf(2)" in message.lower()

    ast_c = make_ast(3, [("CNOT", (0, 1)), ("CNOT", (0, 2))])
    different, diff_message = verifier.equivalent(ast_a, ast_c)
    assert different is False
    assert "gf(2)" in diff_message.lower()


@pytest.mark.skipif(not verifier._QISKIT_AVAILABLE, reason="qiskit not installed")
def test_statevector_equivalence_handles_non_clifford_circuits():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.t(0)
    qc.sx(0)
    ast = qiskit_io.circuit_from_qiskit(qc)
    ast_copy = qiskit_io.circuit_from_qiskit(qc)

    equal, message = verifier.equivalent(ast, ast_copy)
    assert equal is True
    assert "statevector" in message.lower()

    qc_mismatch = QuantumCircuit(1)
    qc_mismatch.t(0)
    qc_mismatch.x(0)
    ast_mismatch = qiskit_io.circuit_from_qiskit(qc_mismatch)

    different, diff_message = verifier.equivalent(ast, ast_mismatch)
    assert different is False
    assert "statevector" in diff_message.lower()


def test_equivalent_detects_qubit_count_mismatch():
    ast_a = make_ast(1, [("H", (0,))])
    ast_b = make_ast(2, [("H", (0,))])

    equal, message = verifier.equivalent(ast_a, ast_b)
    assert equal is False
    assert "qubits" in message.lower()
