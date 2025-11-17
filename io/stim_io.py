"""Conversion helpers between Stim circuits and the internal AST."""
from __future__ import annotations

import importlib.util
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Sequence,
    Union,
    cast,
)

from quirin.core.ast import CircuitAST, Gate

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from stim import Circuit as StimCircuit
    from stim import CircuitInstruction as StimCircuitInstruction
    from stim import CircuitRepeatBlock as StimCircuitRepeatBlock
    from stim import GateTarget as StimGateTarget
else:
    StimCircuit = Any
    StimCircuitInstruction = Any
    StimCircuitRepeatBlock = Any
    StimGateTarget = Any

_STIM_AVAILABLE = importlib.util.find_spec("stim") is not None
if _STIM_AVAILABLE:
    import stim  # type: ignore[import-not-found]
else:  # pragma: no cover - runtime guard when stim missing
    stim = cast("Any", None)

TargetLike = Union[int, StimGateTarget]

SerializedTarget = Dict[str, Any]
SerializedInstruction = Dict[str, Any]

_STIM_TO_AST_NAME = {
    "CX": "CNOT",
}
_AST_TO_STIM_NAME = {v: k for k, v in _STIM_TO_AST_NAME.items()}


def _require_stim() -> None:
    if not _STIM_AVAILABLE:
        raise RuntimeError("stim package not found. Install with `pip install stim`")


def load_stim_text(text: str) -> CircuitAST:
    """Parse a raw .stim string into a CircuitAST."""
    _require_stim()
    circ = stim.Circuit(text)
    return circuit_from_stim(circ)


def load_stim_file(path: str) -> CircuitAST:
    """Parse a .stim file into a CircuitAST."""
    _require_stim()
    circ = stim.Circuit.from_file(path)
    return circuit_from_stim(circ)


def circuit_from_stim(circ: StimCircuit) -> CircuitAST:
    """Convert a stim.Circuit into the internal AST representation."""
    instructions = list(_flatten_instructions(_circuit_items(circ)))
    n_qubits = _infer_qubit_count(instructions)
    ast = CircuitAST(n_qubits)
    for inst in instructions:
        for gate in _convert_instruction(inst):
            ast.append(gate)
    return ast


def circuit_to_stim(circ_ast: CircuitAST) -> StimCircuit:
    """Serialize a CircuitAST back into a stim.Circuit."""
    _require_stim()
    circuit = stim.Circuit()
    for gate in circ_ast.gates:
        if "stim_raw" in gate.params:
            _emit_raw_instruction(circuit, gate.params["stim_raw"])
            continue

        stim_name = _AST_TO_STIM_NAME.get(gate.kind.upper(), gate.kind.upper())
        targets = _targets_for_export(gate)
        args = gate.params.get("stim_args")
        tag = gate.params.get("stim_tag", "")
        if args is None:
            circuit.append(stim_name, targets, (), tag=tag)
        else:
            circuit.append(stim_name, targets, args, tag=tag)
    return circuit


def _circuit_items(
    circ: StimCircuit,
) -> Iterator[Union[StimCircuitInstruction, StimCircuitRepeatBlock]]:
    for idx in range(len(circ)):
        yield circ[idx]


def _flatten_instructions(
    obj: Iterable[Union[StimCircuitInstruction, StimCircuitRepeatBlock]],
) -> Iterator[StimCircuitInstruction]:
    for item in obj:
        if isinstance(item, stim.CircuitInstruction):
            yield cast(StimCircuitInstruction, item)
        elif isinstance(item, stim.CircuitRepeatBlock):
            block = cast(StimCircuitRepeatBlock, item)
            body = block.body_copy()
            for _ in range(block.repeat_count):
                yield from _flatten_instructions(_circuit_items(body))
        else:
            raise TypeError(f"Unsupported stim object {type(item)!r} encountered")


def _infer_qubit_count(instructions: Sequence[StimCircuitInstruction]) -> int:
    max_idx = -1
    for inst in instructions:
        for target in inst.targets_copy():
            if getattr(target, "is_qubit_target", False):
                max_idx = max(max_idx, int(target.value))
    return max_idx + 1 if max_idx >= 0 else 0


def _convert_instruction(inst: StimCircuitInstruction) -> List[Gate]:
    name = inst.name.upper()
    groups = inst.target_groups()
    base_params = _base_params(inst)
    if groups and all(all(t.is_qubit_target for t in group) for group in groups):
        gates: List[Gate] = []
        for group in groups:
            qubits = tuple(int(t.value) for t in group)
            params = dict(base_params)
            params["stim_target_spec"] = [_serialize_target(t) for t in group]
            gate_name = _STIM_TO_AST_NAME.get(name) or name
            gates.append(Gate(gate_name, qubits, params))
        return gates
    return [_gate_from_raw(inst, base_params)]


def _base_params(inst: StimCircuitInstruction) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    args = list(inst.gate_args_copy())
    if args:
        params["stim_args"] = args
    tag = getattr(inst, "tag", "")
    if tag:
        params["stim_tag"] = tag
    return params


def _gate_from_raw(inst: StimCircuitInstruction, base_params: Dict[str, Any]) -> Gate:
    targets = inst.targets_copy()
    serialized: SerializedInstruction = {
        "name": inst.name,
        "targets": [_serialize_target(t) for t in targets],
    }
    args = list(inst.gate_args_copy())
    if args:
        serialized["args"] = args
    tag = getattr(inst, "tag", "")
    if tag:
        serialized["tag"] = tag
    params = dict(base_params)
    params["stim_raw"] = serialized
    qubits = tuple(int(t.value) for t in targets if getattr(t, "is_qubit_target", False))
    return Gate(inst.name.upper(), qubits, params)


def _serialize_target(target: StimGateTarget) -> SerializedTarget:
    if target.is_combiner:
        return {"type": "combiner"}
    if getattr(target, "is_x_target", False):
        return _serialize_pauli_target(target, "X")
    if getattr(target, "is_y_target", False):
        return _serialize_pauli_target(target, "Y")
    if getattr(target, "is_z_target", False):
        return _serialize_pauli_target(target, "Z")
    if target.is_measurement_record_target:
        return {"type": "rec", "value": int(target.value)}
    if target.is_sweep_bit_target:
        return {"type": "sweep", "value": int(target.value)}
    if target.is_qubit_target:
        data: SerializedTarget = {"type": "qubit", "value": int(target.value)}
        if target.is_inverted_result_target:
            data["invert_result"] = True
        return data
    raise ValueError(f"Unsupported stim target type: {target}")


def _serialize_pauli_target(target: StimGateTarget, pauli: str) -> SerializedTarget:
    data: SerializedTarget = {"type": "pauli", "pauli": pauli, "value": int(target.value)}
    if target.is_inverted_result_target:
        data["invert_result"] = True
    return data


def _deserialize_target(data: SerializedTarget) -> Any:
    kind = data.get("type")
    if kind == "combiner":
        return stim.target_combiner()
    if kind == "rec":
        return stim.target_rec(int(data["value"]))
    if kind == "sweep":
        return stim.target_sweep_bit(int(data["value"]))
    if kind == "pauli":
        value = int(data["value"])
        pauli = data["pauli"].upper()
        invert = bool(data.get("invert_result"))
        mapper = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}
        return mapper[pauli](value, invert=invert)
    if kind == "qubit":
        value = int(data["value"])
        if data.get("invert_result"):
            return stim.target_inv(value)
        return value
    raise ValueError(f"Cannot deserialize stim target spec: {data}")


def _targets_for_export(gate: Gate) -> List[TargetLike]:
    spec = gate.params.get("stim_target_spec")
    if spec:
        return [_deserialize_target(entry) for entry in spec]
    return list(gate.qubits)


def _emit_raw_instruction(circuit: StimCircuit, serialized: SerializedInstruction) -> None:
    name = serialized["name"]
    targets = [_deserialize_target(t) for t in serialized.get("targets", [])]
    args = serialized.get("args")
    tag = serialized.get("tag", "")
    if args is None:
        circuit.append(name, targets, (), tag=tag)
    else:
        circuit.append(name, targets, args, tag=tag)
