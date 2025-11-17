# quirin/synth/stub.py
from __future__ import annotations
from typing import List, Tuple

from quirin.core.gf2 import BinaryMatrix
from quirin.synth.api import RowOpSeq, Synthesizer

class StubSynth(Synthesizer):
    """Placeholder synthesizer that leaves the circuit unchanged."""

    def synthesize(self, T: BinaryMatrix, timeout: float | None = None) -> RowOpSeq:  # noqa: ARG002
        if not T.is_identity():
            raise RuntimeError("StubSynth can only handle identity transforms")
        return []
