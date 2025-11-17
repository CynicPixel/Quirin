"""Paper's greedy CNOT synthesizer implementation."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from quirin.core.gf2 import BinaryMatrix
from quirin.synth.api import RowOpSeq, Synthesizer


def _reduce_to_identity(matrix: BinaryMatrix, verbose: bool = False) -> List[Tuple[int, int]]:
    operations: List[Tuple[int, int]] = []
    iteration = 0

    while not matrix.is_identity():
        iteration += 1

        if verbose:
            print(f"Iteration {iteration}:")
            matrix.print_matrix()

        possible_ops = matrix.find_pivot_operations()
        if not possible_ops:
            if verbose:
                print("No more operations possible!")
            break

        best_op = None
        best_benefit = -1

        high_benefit_ops: List[Tuple[int, int, int]] = []
        for source, target in possible_ops:
            benefit = matrix.evaluate_operation_benefit(source, target)
            if benefit >= 2:
                high_benefit_ops.append((source, target, benefit))

        if high_benefit_ops:
            best_source, best_target, best_benefit = max(
                high_benefit_ops, key=lambda item: item[2]
            )
            best_op = (best_source, best_target)

            if verbose:
                print(
                    f"High benefit operation found: {best_op} (benefit: {best_benefit})"
                )
        else:
            for source, target in possible_ops:
                benefit = matrix.evaluate_operation_benefit(source, target)
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_op = (source, target)

            if verbose and best_op is not None:
                print(f"Best available operation: {best_op} (benefit: {best_benefit})")

        if best_op is None:
            if verbose:
                print("No beneficial operation found!")
            break

        cnot_gate = matrix.apply_row_operation(best_op[0], best_op[1])
        operations.append(cnot_gate)

        if verbose:
            print(
                f"Applied operation: R{best_op[1]} = R{best_op[0]} + R{best_op[1]}"
            )
            print(f"Equivalent CNOT: {cnot_gate}")

    if verbose:
        print(f"Reduction complete in {iteration} iterations")
        print(f"Final matrix is identity: {matrix.is_identity()}")
        matrix.print_matrix()

    return operations


class PaperGreedyOptimizer:
    """Standalone optimizer that mirrors Code/paper_greedy behavior."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.stats: Dict[str, int] = {}

    def optimize(
        self, n_qubits: int, gates: Sequence[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
        matrix = BinaryMatrix(n_qubits)

        if self.verbose:
            print("Building transformed matrix...")

        for control, target in gates:
            matrix.apply_cnot(control, target)
            if self.verbose:
                print(f"Applied CNOT({control}, {target})")
                matrix.print_matrix()

        initial_ones = matrix.count_ones()

        if self.verbose:
            print(f"Transformed matrix has {initial_ones} ones")
            matrix.print_matrix()

        reduction_operations = _reduce_to_identity(matrix, verbose=self.verbose)
        optimized_gates = list(reversed(reduction_operations))

        self.stats = {
            "initial_ones": initial_ones,
            "final_ones": 0 if matrix.is_identity() else matrix.count_ones(),
            "reduction_operations": len(reduction_operations),
            "original_gates": len(gates),
            "optimized_gates": len(optimized_gates),
            "improvement": len(gates) - len(optimized_gates),
        }

        return optimized_gates, self.stats

    def get_statistics(self) -> Dict[str, int]:
        return self.stats.copy()

    def _find_reduction_operations(self, matrix: BinaryMatrix) -> List[Tuple[int, int]]:
        return _reduce_to_identity(matrix, verbose=self.verbose)


class PaperGreedySynth(Synthesizer):
    """Synthesizer adapter exposing paper_greedy via the Synthesizer API."""

    def __init__(self, verbose: bool = False) -> None:
        self.optimizer = PaperGreedyOptimizer(verbose=verbose)

    def synthesize(self, T: BinaryMatrix, timeout: float | None = None) -> RowOpSeq:  # noqa: ARG002
        matrix = T.clone()
        reduction_ops = self.optimizer._find_reduction_operations(matrix)

        if not matrix.is_identity():
            raise RuntimeError("paper_greedy failed to reduce matrix to identity")

        optimized = list(reversed(reduction_ops))
        return optimized
