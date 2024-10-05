import os
import json
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from src.VQC_ import VariationalQuantumClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from src.ansatz import *

LOG_DIR = "results_2"

os.makedirs(LOG_DIR, exist_ok=True)

class Experiment:
    def __init__(self, num_qubits, sample_size=100, experiment_name: Optional[str] = None):
        self.sample_size = sample_size
        self.num_qubits = num_qubits
        self.validation_results = []
        self.experiment_name = experiment_name
        self.train_results = {}

    def split_data(self, X, y, use_sample: bool = True, train_size=0.7, val_size=0.2, test_size=0.1, verbose=False):
        """Splits the data into training, validation, and test sets. Allows training on a sample if use_sample is True."""
        self._X, self._y = X, y
        
        if use_sample:
            self.X = copy.deepcopy(self._X[:self.sample_size])
            self.y = copy.deepcopy(self._y[:self.sample_size])
        else:
            self.X = copy.deepcopy(self._X)
            self.y = copy.deepcopy(self._y)
            
        temp_size = val_size + test_size
        sss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=temp_size, random_state=42)

        for train_index, temp_index in sss_train_test.split(self.X, self.y):
            X_train, X_temp = self.X[train_index], self.X[temp_index]
            y_train, y_temp = self.y[train_index], self.y[temp_index]

        val_test_ratio = val_size / (val_size + test_size)
        sss_valid_test = StratifiedShuffleSplit(n_splits=1, test_size=(1 - val_test_ratio), random_state=42)

        for val_index, test_index in sss_valid_test.split(X_temp, y_temp):
            X_val, X_test = X_temp[val_index], X_temp[test_index]
            y_val, y_test = y_temp[val_index], y_temp[test_index]

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        if verbose:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.hist(y_train, bins=2)
            plt.title('Train set')
            plt.subplot(1, 3, 2)
            plt.hist(y_val, bins=2)
            plt.title('Validation set')
            plt.subplot(1, 3, 3)
            plt.hist(y_test, bins=2)
            plt.title('Test set')
            plt.show()
        
        print(f"[-] Data split into {len(X_train)} training samples, {len(X_val)} validation samples, and {len(X_test)} test samples")

    def run_experiment(self, feature_map, ansatz_funcs, max_reps=1, verbose=False):
        self.feature_map = feature_map
        self.asnatx_funcs = ansatz_funcs
        for i in range(len(ansatz_funcs)):
            for reps in range(1, max_reps + 1):
                if os.path.exists(f"./{LOG_DIR}/validation_results_{self.feature_map._base_name}_{self.experiment_name}.json"):
                    with open(f"./{LOG_DIR}/validation_results_{self.feature_map._base_name}_{self.experiment_name}.json", "r") as f:
                        self.validation_results = json.load(f)
                
                if any([res['ansatz'] == ansatz_funcs[i].__name__ and res['reps'] == reps and res['feature_map'] == feature_map._base_name for res in self.validation_results]):
                    print(f"   [>] Skipping training for {self.feature_map._base_name}, {ansatz_funcs[i].__name__} with {reps} repetitions")
                    continue
                
                print(f"[+] Training {ansatz_funcs[i].__name__} with {reps} repetitions")
                start = time.time()
                vqc = VariationalQuantumClassifier(self.feature_map, self.num_qubits, ansatz_funcs[i])
                
                vqc.train(self.X_train, self.y_train, reps=reps, verbose=verbose)

                accuracy, precision, recall, f1 = vqc.evaluate(self.X_val, self.y_val)
                self.validation_results.append({
                    "ansatz": ansatz_funcs[i].__name__,
                    "feature_map": self.feature_map._base_name,
                    "reps": reps,
                    "accuracy": round(accuracy, 3),
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "f1": round(f1, 3),
                    "time": round(time.time() - start, 3)
                })
                print(f"[+] Validation results for {ansatz_funcs[i].__name__} with {reps} repetitions:")
                print(f"   [>] Accuracy: {accuracy}")
                
                with open(f"./{LOG_DIR}/validation_results_{self.feature_map._base_name}_{self.experiment_name}.json", "w") as f:
                    json.dump(self.validation_results, f, indent=4, default=str)
            
    def evaluate_best_on_test(self, use_full_data=False):
        if use_full_data:
            assert self.X is not None and self.y is not None, "No data available to evaluate on full dataset"
            self.split_data(self._X, self._y, use_sample=False, verbose=True)
        
        best_model = max(self.validation_results, key=lambda x: x["accuracy"])
        print(f"[-] Evaluating best model on test set: {best_model}")
        for asnatx in self.asnatx_funcs:
            if asnatx.__name__ == best_model["ansatz"]:
                best_asnatz = asnatx
                break
        
        start = time.time()
        vqc = VariationalQuantumClassifier(self.feature_map, self.num_qubits, best_asnatz)
        vqc.train(self.X_train, self.y_train, reps=best_model["reps"])
        
        accuracy, precision, recall, f1 = vqc.evaluate(self.X_test, self.y_test)
        self.test_results = {
            "ansatz": best_model['ansatz'],
            "feature_map": best_model['feature_map'],
            "reps": best_model['reps'],
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall,3),
            "f1": round(f1, 3),
            "time": round(time.time() - start, 3)
        }
        
        with open(f"./{LOG_DIR}/test_results_{self.feature_map._base_name}_{self.experiment_name}.json", "w") as f:
            json.dump(self.train_results, f, indent=4)
        return self.test_results
            
    def plot_results(self):
        """Plots the validation results."""
        ansatz_names = [f"{res['ansatz']}_{res['reps']}" for res in self.validation_results]
        accuracies = [res['accuracy'] for res in self.validation_results]
        precisions = [res['precision'] for res in self.validation_results]
        recalls = [res['recall'] for res in self.validation_results]
        f1s = [res['f1'] for res in self.validation_results]
        
        plt.figure(figsize=(15, 8))
        bar_width = 0.2
        index = np.arange(len(ansatz_names))
        plt.bar(index, accuracies, bar_width, label="Accuracy")
        plt.bar(index + bar_width, precisions, bar_width, label="Precision")
        plt.bar(index + 2 * bar_width, recalls, bar_width, label="Recall")
        plt.bar(index + 3 * bar_width, f1s, bar_width, label="F1")
        plt.xlabel("Ansatz")
        plt.ylabel("Score")
        plt.title("Validation Results")
        plt.xticks(index + bar_width, ansatz_names)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"./{LOG_DIR}/validation_results_{self.feature_map._base_name}_{self.experiment_name}.png")
        plt.show()
