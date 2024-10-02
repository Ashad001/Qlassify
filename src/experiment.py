
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.VQC_ import VariationalQuantumClassifier

os.makedirs("results", exist_ok=True)

class Experiment:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, num_qubits):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.num_qubits = num_qubits
        self.validation_results = []
        self.train_results = {}
        
    def run_experiment(self, feature_map, ansatz_funcs, max_reps = 1, verbose=False):
        for i in range(len(ansatz_funcs)):
            for reps in range(1, max_reps + 1):
                with open("./results/validation_results.json", "r") as f:
                    self.validation_results = json.load(f)
                if any([res['ansatz'] == ansatz_funcs[i].__name__ and res['reps'] == reps for res in self.validation_results]):
                    continue
                vqc = VariationalQuantumClassifier(feature_map, self.num_qubits, ansatz_funcs[i])
                vqc.train(self.X_train, self.y_train, reps=reps, verbose=verbose)

                accuracy, precision, recall, f1 = vqc.evaluate(self.X_val, self.y_val)
                self.validation_results.append({
                    "ansatz": ansatz_funcs[i].__name__,
                    "reps": reps,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "model": vqc.model
                })
                print(f"[+] Validation results for {ansatz_funcs[i].__name__} with {reps} repetitions:")
                print(f"[+] Accuracy: {accuracy}")
                # Why save here you ask???? Well you want me to run the experiment again??
                with open("./results/validation_results.json", "w") as f:
                    json.dump(self.validation_results, f)
            
    def evaluate_best_on_test(self):
        best_model = max(self.validation_results, key=lambda x: x["accuracy"])["model"]
        accuracy, precision, recall, f1 = best_model.evaluate(self.X_test, self.y_test)
        self.train_results = {
            "ansatz": best_model['ansatz'],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        with open("./results/train_results.json", "w") as f:
            json.dump(self.train_results, f)
            
    def plot_results(self):
        ansatx_names = [res['ansatz'] for res in self.validation_results]
        accuracies = [res['accuracy'] for res in self.validation_results]
        precisions = [res['precision'] for res in self.validation_results]
        recalls = [res['recall'] for res in self.validation_results]
        f1s = [res['f1'] for res in self.validation_results]
        
        plt.figure(figsize=(10, 6))
        bar_width = 0.2
        index = np.arrange(len(ansatx_names))
        plt.bar(index, accuracies, bar_width, label="Accuracy")
        plt.bar(index + bar_width, precisions, bar_width, label="Precision")
        plt.bar(index + 2 * bar_width, recalls, bar_width, label="Recall")
        plt.bar(index + 3 * bar_width, f1s, bar_width, label="F1")
        plt.xlabel("Ansatz")
        plt.ylabel("Score")
        plt.title("Validation Results")
        plt.xticks(index + bar_width, ansatx_names)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("./results/validation_results.png")
        plt.show()