import numpy as np
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.callbacks import callback_graph    

class VariationalQuantumClassifier:
    def __init__(self, feature_map, num_qbits, ansantx_fn, optimizer=COBYLA(maxiter=25)):
        self.feature_map = feature_map
        self.num_qbits = num_qbits
        self.ansantx_fn = ansantx_fn
        self.optimizer = optimizer
        self.model = None

    def build_model(self, reps = 1, verbose = False):
        ansantz = self.ansantx_fn(self.num_qbits, reps)
        if verbose:
            print(f"[+] Building model with {self.num_qbits} qubits and {reps} repetitions")
            self.vqc = self.model = VQC(
                feature_map=self.feature_map, 
                ansatz=ansantz, 
                optimizer=self.optimizer,
                callback=callback_graph
            )
        else:
            self.vqc =  self.model = VQC(
                feature_map=self.feature_map, 
                ansatz=ansantz, 
                optimizer=self.optimizer
            )
    def train(self, X_train, y_train, reps = 1, verbose = False):
        if self.model is None:
            self.build_model(reps=reps, verbose=verbose)
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_val, y_val):
        if self.model is None:
            raise Exception("Model not trained")
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        return accuracy, precision, recall, f1
        
        