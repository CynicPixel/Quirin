"""Qiskit ↔ CircuitAST conversion helpers."""

from __future__ import annotations

import copy
import importlib.util
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

from quirin.core.ast import CircuitAST, Gate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qiskit import QuantumCircuit
    from qiskit.circuit import ClassicalRegister, QuantumRegister
else:  # pragma: no cover - runtime when qiskit missing
    QuantumCircuit = Any
    ClassicalRegister = Any
    QuantumRegister = Any


_QISKIT_AVAILABLE = importlib.util.find_spec("qiskit") is not None

_QISKIT_TO_AST_NAME = {
    "CX": "CNOT",
    "CNOT": "CNOT",
}
_AST_TO_QISKIT_NAME = {"CNOT": "cx", "H": "h", "CZ": "cz", "S": "s", "X": "x", "Z": "z"}


def is_qiskit_available() -> bool:
    """Return True when qiskit can be imported."""

    return _QISKIT_AVAILABLE


def circuit_from_qiskit(qc: "QuantumCircuit") -> CircuitAST:
    """Convert a :class:`qiskit.QuantumCircuit` into a :class:`CircuitAST`."""

    _require_qiskit()
    qubit_map = {qubit: idx for idx, qubit in enumerate(qc.qubits)}
    clbit_map = {clbit: idx for idx, clbit in enumerate(qc.clbits)}

    ast = CircuitAST(qc.num_qubits)
    ast.qiskit_metadata = _extract_circuit_metadata(qc)  # type: ignore[attr-defined]

    instructions: Iterable[CircuitInstruction] = qc.data  # type: ignore[assignment]
    for inst in instructions:
        gate = _instruction_to_gate(inst, qubit_map, clbit_map)
        ast.append(gate)
    return ast


def circuit_to_qiskit(ast: CircuitAST) -> "QuantumCircuit":
    """Serialize a :class:`CircuitAST` back into Qiskit's :class:`QuantumCircuit`."""

    _require_qiskit()
    metadata: Dict[str, Any] = getattr(ast, "qiskit_metadata", {}) or {}
    circuit = _build_qiskit_circuit(ast, metadata)
    for gate in ast.gates:
        _append_gate_to_circuit(circuit, gate)
    _apply_metadata(circuit, metadata)
    return circuit


def _require_qiskit() -> None:
    if not _QISKIT_AVAILABLE:
        raise RuntimeError("qiskit not installed; install with `pip install qiskit`")


def _instruction_to_gate(
    inst: Any,
    qubit_map: Dict[Any, int],
    clbit_map: Dict[Any, int],
) -> Gate:
    operation = inst.operation
    name = operation.name.upper()
    ast_name = _QISKIT_TO_AST_NAME.get(name) or name
    qubits = tuple(qubit_map[qubit] for qubit in inst.qubits)
    params = _instruction_params(operation, inst.clbits, clbit_map)
    return Gate(ast_name, qubits, params)


def _instruction_params(operation: Any, clbits: Sequence[Any], clbit_map: Dict[Any, int]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    params["qiskit_operation"] = _clone_operation(operation)
    if operation.params:
        params["qiskit_params"] = list(operation.params)
    label = getattr(operation, "label", None)
    if label is not None:
        params["label"] = label
    condition = getattr(operation, "_condition", None)
    if condition is not None:
        params["condition"] = condition
    if clbits:
        params["qiskit_clbits"] = tuple(clbit_map[clbit] for clbit in clbits)
    return params


def _clone_operation(operation: Any) -> Any:
    copier = getattr(operation, "copy", None)
    if callable(copier):
        return copier()
    return copy.deepcopy(operation)


def _extract_circuit_metadata(qc: "QuantumCircuit") -> Dict[str, Any]:
    classical_regs = [
        {"name": reg.name, "size": len(reg)}
        for reg in getattr(qc, "cregs", [])
    ]
    quantum_regs = [
        {"name": reg.name, "size": len(reg)}
        for reg in getattr(qc, "qregs", [])
    ]
    metadata: Dict[str, Any] = {
        "name": qc.name,
        "metadata": copy.deepcopy(getattr(qc, "metadata", None)),
        "global_phase": getattr(qc, "global_phase", 0),
        "num_clbits": qc.num_clbits,
        "classical_registers": classical_regs,
        "quantum_registers": quantum_regs,
    }
    calibrations = getattr(qc, "calibrations", None)
    if calibrations:
        metadata["calibrations"] = copy.deepcopy(calibrations)
    return metadata


def _build_qiskit_circuit(ast: CircuitAST, metadata: Dict[str, Any]) -> "QuantumCircuit":
    from qiskit import QuantumCircuit as _RuntimeQuantumCircuit  # type: ignore
    from qiskit.circuit import ClassicalRegister, QuantumRegister  # type: ignore

    quantum_regs = metadata.get("quantum_registers") or []
    classical_regs = metadata.get("classical_registers") or []
    regs: List[Any] = []

    total_qubits = sum(int(reg.get("size", 0)) for reg in quantum_regs)
    if total_qubits == ast.n_qubits and quantum_regs:
        for reg in quantum_regs:
            regs.append(QuantumRegister(int(reg["size"]), name=reg.get("name")))
    else:
        regs.append(QuantumRegister(ast.n_qubits))

    inferred_clbits = _infer_required_clbits(ast)
    explicit_clbits = sum(int(reg.get("size", 0)) for reg in classical_regs)
    target_clbits = max(metadata.get("num_clbits", 0), inferred_clbits, explicit_clbits)

    if classical_regs and explicit_clbits >= target_clbits:
        for reg in classical_regs:
            regs.append(ClassicalRegister(int(reg["size"]), name=reg.get("name")))
    elif target_clbits:
        regs.append(ClassicalRegister(target_clbits))

    name = metadata.get("name")
    global_phase = metadata.get("global_phase", 0)
    meta_payload = copy.deepcopy(metadata.get("metadata")) if metadata.get("metadata") else None
    circuit = _RuntimeQuantumCircuit(*regs, name=name, global_phase=global_phase, metadata=meta_payload)
    return circuit


def _infer_required_clbits(ast: CircuitAST) -> int:
    max_idx = -1
    for gate in ast.gates:
        clbits: Optional[Sequence[int]] = gate.params.get("qiskit_clbits")
        if clbits:
            max_idx = max(max_idx, max(clbits))
    return max_idx + 1 if max_idx >= 0 else 0


def _append_gate_to_circuit(circuit: "QuantumCircuit", gate: Gate) -> None:
    operation = gate.params.get("qiskit_operation")
    clbits = gate.params.get("qiskit_clbits")
    qubits = [circuit.qubits[idx] for idx in gate.qubits]
    cl_args = [circuit.clbits[idx] for idx in clbits] if clbits else ()
    if operation is not None:
        circuit.append(_clone_operation(operation), qubits, cl_args)
        return
    _append_fallback_gate(circuit, gate, qubits, cl_args)


def _append_fallback_gate(
    circuit: "QuantumCircuit",
    gate: Gate,
    qubits: Sequence[Any],
    cl_args: Sequence[Any],
) -> None:
    name = gate.kind.upper()
    if name in _AST_TO_QISKIT_NAME:
        method = getattr(circuit, _AST_TO_QISKIT_NAME[name])
        method(*gate.qubits)
        return
    if name == "MEASURE":
        if not cl_args or len(cl_args) != len(qubits):
            raise ValueError("Measurement gate missing classical bit mapping")
        for qubit, clbit in zip(qubits, cl_args):
            circuit.measure(qubit, clbit)
        return
    raise ValueError(f"Cannot export gate kind '{gate.kind}' without qiskit metadata")


def _apply_metadata(circuit: "QuantumCircuit", metadata: Dict[str, Any]) -> None:
    if "calibrations" in metadata:
        setattr(circuit, "calibrations", copy.deepcopy(metadata["calibrations"]))
