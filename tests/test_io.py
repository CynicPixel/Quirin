"""Regression tests for the stim, Qiskit, and OpenQASM IO helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quirin.io import qasm_io, qiskit_io, stim_io  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore::qiskit.exceptions.ExperimentalWarning"),
    pytest.mark.filterwarnings("ignore:This is an experimental native version of the OpenQASM 3 importer"),
]


def _build_sample_qiskit_circuit():
    qiskit = pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

    qreg = QuantumRegister(2, "qr")
    creg = ClassicalRegister(2, "cr")
    qc = QuantumCircuit(qreg, creg, name="io-sample", metadata={"suite": "io"})
    qc.global_phase = 0.25
    qc.h(qreg[0])
    qc.cx(qreg[0], qreg[1])
    qc.measure(qreg, creg)
    return qc


def _instruction_signature(circuit):
    def bit_indices(bits):
        return [circuit.find_bit(bit).index for bit in bits]

    return [
        (instruction.operation.name, bit_indices(instruction.qubits), bit_indices(instruction.clbits))
        for instruction in circuit.data
    ]


def test_stim_round_trip_preserves_text():
    stim = pytest.importorskip("stim")

    program = """
    H 0
    CX 0 1
    TICK
    M 0 1
    """
    ast = stim_io.load_stim_text(program)
    rebuilt = stim_io.circuit_to_stim(ast)

    assert str(stim.Circuit(program)).strip() == str(rebuilt).strip()


def test_qiskit_round_trip_preserves_metadata():
    qc = _build_sample_qiskit_circuit()
    ast = qiskit_io.circuit_from_qiskit(qc)

    assert ast.n_qubits == qc.num_qubits
    assert ast.gates[1].kind == "CNOT"

    rebuilt = qiskit_io.circuit_to_qiskit(ast)

    assert rebuilt.name == qc.name
    assert rebuilt.metadata == qc.metadata
    assert rebuilt.num_clbits == qc.num_clbits
    assert rebuilt.global_phase == pytest.approx(qc.global_phase)
    assert _instruction_signature(rebuilt) == _instruction_signature(qc)


def test_qasm_string_round_trip(tmp_path):
    qc = _build_sample_qiskit_circuit()
    ast = qiskit_io.circuit_from_qiskit(qc)

    qasm_text = qasm_io.dumps_qasm3(ast)
    assert "OPENQASM 3.0" in qasm_text

    try:
        loaded_ast = qasm_io.loads_qasm3(qasm_text, experimental=True)
    except RuntimeError as exc:  # pragma: no cover - optional extra fallback
        if "qasm3-import" in str(exc):
            pytest.skip("qasm3 importer extra not installed")
        raise

    rebuilt = qiskit_io.circuit_to_qiskit(loaded_ast)
    assert _instruction_signature(rebuilt) == _instruction_signature(qc)

    output_path = tmp_path / "program.qasm"
    qasm_io.dump_qasm3_file(ast, output_path)
    try:
        loaded_from_file = qasm_io.load_qasm3_file(str(output_path), experimental=True)
    except RuntimeError as exc:  # pragma: no cover - optional extra fallback
        if "qasm3-import" in str(exc):
            pytest.skip("qasm3 importer extra not installed")
        raise

    rebuilt_from_file = qiskit_io.circuit_to_qiskit(loaded_from_file)
    assert _instruction_signature(rebuilt_from_file) == _instruction_signature(qc)
