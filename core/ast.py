# quirin/core/ast.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Iterable

# Canonical 0-indexed internal circuit representation.

@dataclass
class QubitMeta:
    idx: int               # 0-index
    name: Optional[str] = None
    init_state: Optional[str] = None  # '0','1','+','-','unknown' or None
    ancilla: bool = False
    # frame can help track logical basis for safe rewrites
    frame: str = 'Z'       # 'Z' or 'X' or other label

    def __post_init__(self):
        if self.name is None:
            self.name = f"q{self.idx}"


@dataclass
class Gate:
    kind: str                     # 'CNOT','CZ','H','S','X','Z','CY','MEASURE', ...
    qubits: Tuple[int, ...]       # tuple of 0-index ints
    params: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"Gate({self.kind}, {self.qubits}, {self.params})"


@dataclass
class CircuitAST:
    n_qubits: int
    gates: List[Gate] = field(default_factory=list)
    qubits_meta: List[QubitMeta] = field(init=False)

    def __post_init__(self):
        self.qubits_meta = [QubitMeta(i) for i in range(self.n_qubits)]

    def append(self, gate: Gate):
        self.gates.append(gate)

    def extend(self, gates: Iterable[Gate]):
        self.gates.extend(gates)

    def insert(self, index: int, gate: Gate):
        self.gates.insert(index, gate)

    def copy(self) -> "CircuitAST":
        # shallow copy of gates and deep-ish metadata
        new = CircuitAST(self.n_qubits)
        new.gates = [Gate(g.kind, tuple(g.qubits), dict(g.params)) for g in self.gates]
        new.qubits_meta = [QubitMeta(q.idx, q.name, q.init_state, q.ancilla, q.frame)
                           for q in self.qubits_meta]
        return new

    def to_cnot_blocks(self):
        """
        Extract contiguous segments of gates that are either CNOT or convertible-to-CNOT.
        For now, consider gates kind == 'CNOT' only. Later the pass manager will
        convert CZ/CY -> CNOT+single-qubit gates.
        Returns list of (start_index, end_index) where end_index is exclusive.
        """
        blocks = []
        i = 0
        while i < len(self.gates):
            if self.gates[i].kind == "CNOT":
                j = i + 1
                while j < len(self.gates) and self.gates[j].kind == "CNOT":
                    j += 1
                blocks.append((i, j))
                i = j
            else:
                i += 1
        return blocks

    def replace_block(self, start: int, end: int, new_gates: List[Gate]):
        """Replace gates[start:end] with new_gates."""
        self.gates[start:end] = new_gates

    def __len__(self):
        return len(self.gates)
