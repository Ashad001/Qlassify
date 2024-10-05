import qiskit
from qiskit.circuit import QuantumCircuit
from typing import Type

class FeatureMapGenerator:
    def __init__(self, feature_map_class: Type[QuantumCircuit], feature_dimension: int, reps: int = 2, entanglement: str = "full") -> None:
        self.feature_map_class = feature_map_class
        self.feature_dimension = feature_dimension
        self.reps = reps
        self.entanglement = entanglement

    def get_feature_map(self) -> QuantumCircuit:
        """
        Returns the feature map based on the called map (PauliFeatureMap or ZZFeatureMap).
        """
        return self.feature_map_class(
            feature_dimension=self.feature_dimension,
            reps=self.reps,
            entanglement=self.entanglement
        )