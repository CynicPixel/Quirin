# quirin/synth/api.py
from __future__ import annotations
from typing import List, Tuple

from quirin.core.gf2 import BinaryMatrix

# RowOp = (src, tgt)
RowOpSeq = List[Tuple[int,int]]

class Synthesizer:
    """Base class for linear reversible (CNOT-only) synthesizers."""

    def synthesize(self, T: BinaryMatrix, timeout: float | None = None) -> RowOpSeq:
        """Return row operations that map the identity transform to `T`.

        `timeout` defaults to ``None`` meaning "run to convergence". Implementations may
        optionally honor positive timeout values but must not impose iteration limits when
        ``timeout`` is unset.
        """

        raise NotImplementedError("Implement in subclass")
