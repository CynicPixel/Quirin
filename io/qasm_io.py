"""OpenQASM 3 helpers built on :mod:`qiskit.qasm3`.

This module mirrors the workflows described in ``qasm.txt`` by providing
light wrapper functions over :mod:`qiskit.qasm3`'s ``dump``/``dumps`` and
``load``/``loads`` entrypoints. Circuits are converted to the internal
``CircuitAST`` via the Qiskit bridge utilities.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Dict, Optional

from quirin.core.ast import CircuitAST
from quirin.io.qiskit_io import circuit_from_qiskit, circuit_to_qiskit

_QASM3_MODULE = "qiskit.qasm3"
_QASM3_AVAILABLE = importlib.util.find_spec(_QASM3_MODULE) is not None
_QASM3_IMPORT_HINT = (
    "OpenQASM 3 import requires the optional qiskit_qasm3_import extra. "
    "Install via `pip install \"qiskit[qasm3-import]\"`."
)


def _require_qasm3_module():
    if not _QASM3_AVAILABLE:
        raise RuntimeError("qiskit.qasm3 is unavailable; install qiskit>=0.45.")
    return importlib.import_module(_QASM3_MODULE)


def load_qasm3_file(
    path: str,
    *,
    num_qubits: Optional[int] = None,
    experimental: bool = False,
    **kwargs: Any,
) -> CircuitAST:
    """Load an OpenQASM 3 file into ``CircuitAST`` via :func:`qiskit.qasm3.load`.

    Args:
        path: Path to the OpenQASM 3 source.
        num_qubits: Optional physical width hint for the legacy loader.
        experimental: When ``True``, use the Rust-based ``load_experimental`` parser.
        **kwargs: Forwarded to the underlying loader (``annotation_handlers``,
            ``custom_gates``, ``include_path``, ...).
    """

    qasm3 = _require_qasm3_module()
    loader = qasm3.load_experimental if experimental else qasm3.load
    loader_kwargs: Dict[str, Any] = dict(kwargs)
    if not experimental and num_qubits is not None:
        loader_kwargs.setdefault("num_qubits", num_qubits)
    try:
        circuit = loader(path, **loader_kwargs)
    except ImportError as exc:  # Raised when qiskit_qasm3_import is missing
        raise RuntimeError(_QASM3_IMPORT_HINT) from exc
    return circuit_from_qiskit(circuit)


def loads_qasm3(
    source: str,
    *,
    num_qubits: Optional[int] = None,
    experimental: bool = False,
    **kwargs: Any,
) -> CircuitAST:
    """Parse an OpenQASM 3 string using :func:`qiskit.qasm3.loads`."""

    qasm3 = _require_qasm3_module()
    loader = qasm3.loads_experimental if experimental else qasm3.loads
    loader_kwargs: Dict[str, Any] = dict(kwargs)
    if not experimental and num_qubits is not None:
        loader_kwargs.setdefault("num_qubits", num_qubits)
    try:
        circuit = loader(source, **loader_kwargs)
    except ImportError as exc:
        raise RuntimeError(_QASM3_IMPORT_HINT) from exc
    return circuit_from_qiskit(circuit)


def dump_qasm3_file(ast: CircuitAST, path: str, **kwargs: Any) -> None:
    """Serialize ``CircuitAST`` to a file via :func:`qiskit.qasm3.dump`."""

    qasm3 = _require_qasm3_module()
    circuit = circuit_to_qiskit(ast)
    with open(path, "w", encoding="utf-8") as stream:
        qasm3.dump(circuit, stream, **kwargs)


def dumps_qasm3(ast: CircuitAST, **kwargs: Any) -> str:
    """Return the OpenQASM 3 string for ``ast`` via :func:`qiskit.qasm3.dumps`."""

    qasm3 = _require_qasm3_module()
    circuit = circuit_to_qiskit(ast)
    return qasm3.dumps(circuit, **kwargs)
