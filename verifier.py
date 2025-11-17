# quirin/verifier.py
from __future__ import annotations
from typing import Optional, Tuple
from quirin.core.ast import CircuitAST
from quirin.core.gf2 import BinaryMatrix
import importlib.util

_STIM_AVAILABLE = importlib.util.find_spec("stim") is not None
_QISKIT_AVAILABLE = importlib.util.find_spec("qiskit") is not None

# Primary strategy:
# 1. If Stim available and circuit is Clifford-only, use Stim tableau equality (fast).
# 2. Else if circuits are CNOT-only (can be detected), compare GF(2) matrices.
# 3. Fallback: use Qiskit statevector simulation for small n (n <= threshold).

def is_clifford_only(ast: CircuitAST) -> bool:
    # conservative: only simple cliffords
    for g in ast.gates:
        if g.kind not in ("CNOT","H","S","X","Z","CZ","CY","MEASURE"):
            return False
        # measure is OK but complicates equivalence; for simplicity treat it as non-Clifford
        if g.kind == "MEASURE":
            return False
    return True

def is_cnot_only(ast: CircuitAST) -> bool:
    for g in ast.gates:
        if g.kind != "CNOT":
            return False
    return True

def mat_from_ast_cnot(ast: CircuitAST) -> BinaryMatrix:
    # extract CNOT sequence in order
    seq = [(g.qubits[0], g.qubits[1]) for g in ast.gates if g.kind == "CNOT"]
    return BinaryMatrix.matrix_from_cnot_seq(ast.n_qubits, seq)

def equivalent(ast_a: CircuitAST, ast_b: CircuitAST, sv_threshold: int = 12) -> Tuple[bool, str]:
    if ast_a.n_qubits != ast_b.n_qubits:
        return False, "Circuits use different numbers of qubits"

    if is_cnot_only(ast_a) and is_cnot_only(ast_b):
        return _cnot_matrix_equivalent(ast_a, ast_b)

    stim_result = _stim_tableau_equivalent(ast_a, ast_b)
    if stim_result is not None:
        return stim_result

    sv_result = _statevector_equivalent(ast_a, ast_b, sv_threshold)
    if sv_result is not None:
        return sv_result

    return False, "No verifier available (install stim/qiskit or raise sv_threshold)"


def _stim_tableau_equivalent(ast_a: CircuitAST, ast_b: CircuitAST) -> Optional[Tuple[bool, str]]:
    if not _STIM_AVAILABLE or not (is_clifford_only(ast_a) and is_clifford_only(ast_b)):
        return None
    try:
        import stim
        from quirin.io.stim_io import circuit_to_stim

        circ_a = circuit_to_stim(ast_a)
        circ_b = circuit_to_stim(ast_b)
        tableau_a = stim.Tableau.from_circuit(circ_a)
        tableau_b = stim.Tableau.from_circuit(circ_b)
    except Exception:
        return None

    if tableau_a == tableau_b:
        return True, "Equal (stim tableau)"
    return False, "Differing stim tableau"


def _cnot_matrix_equivalent(ast_a: CircuitAST, ast_b: CircuitAST) -> Tuple[bool, str]:
    mat_a = mat_from_ast_cnot(ast_a)
    mat_b = mat_from_ast_cnot(ast_b)
    equal = mat_a.equals(mat_b)
    if equal:
        return True, "Equal (GF(2) CNOT-only check)"
    return False, "Differ (GF(2) CNOT-only check)"


def _statevector_equivalent(
    ast_a: CircuitAST, ast_b: CircuitAST, sv_threshold: int
) -> Optional[Tuple[bool, str]]:
    if not _QISKIT_AVAILABLE:
        return None
    if ast_a.n_qubits > sv_threshold:
        return None
    if any(g.kind == "MEASURE" for g in ast_a.gates + ast_b.gates):
        return None

    try:
        from qiskit.quantum_info import Statevector
        from quirin.io.qiskit_io import circuit_to_qiskit

        qc_a = circuit_to_qiskit(ast_a)
        qc_b = circuit_to_qiskit(ast_b)
        state_a = Statevector.from_instruction(qc_a)
        state_b = Statevector.from_instruction(qc_b)
    except Exception:
        return None

    if state_a.equiv(state_b):
        return True, "Equal (statevector)"
    return False, "Differ (statevector)"
