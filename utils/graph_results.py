import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os

LOG_DIR = "results_2"

test_results_files = glob.glob(f"./{LOG_DIR}/test_results_*")

if len(test_results_files) > 0:
    for file in test_results_files:
        with open(file, 'r') as f:
            test_results = json.load(f)

        values = [
            test_results.get('accuracy', 0),
            test_results.get('precision', 0),
            test_results.get('recall', 0),
            test_results.get('f1', 0)
        ]

        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        N = len(categories)

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]  # Ensure the radar chart is closed
        angles += angles[:1]

        # Radar plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.fill(angles, values, color='#1f77b4', alpha=0.3)
        ax.plot(angles, values, color='#1f77b4', linewidth=2, marker='o', markersize=6)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=14, fontweight='bold', color='darkblue')

        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=12, color='gray')
        ax.yaxis.grid(True, color='gray', linestyle='--', alpha=0.7)

        ax.spines['polar'].set_visible(True)
        ax.spines['polar'].set_color('gray')
        ax.spines['polar'].set_linewidth(1.5)

        # name = os.path.basename(file)
        name = os.path.basename(file).split('_')[2:]
        name = " ".join(name).split('.')[0].replace("_", ' ').replace("t1", "Heart Disease").replace("pauli", '').replace('zz', '')
        plt.title(f'Test Performance: {name}', size=16, color='darkblue', weight='bold', pad=20)

        plt.tight_layout()
        # plt.show()
        
        plt.savefig(f'./{LOG_DIR}/{name}.png')

else:
    print("No test result files found in the specified directory.")
