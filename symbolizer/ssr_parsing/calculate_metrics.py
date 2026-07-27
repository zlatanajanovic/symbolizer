import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import your metric calculation functions from your codebase
# Update the paths accordingly if needed
from .run_eval_vlm import (
    calculate_metrics,
    calculate_metrics_objects,
    calculate_metrics_per_domain,
    calculate_metrics_objects_per_domain,
)

RESULTS_FOLDER = "./results/gpt_v/pddlgym_own_results/"
OBJECTS_FILE = "objects_evaluation_results.jsonl"
ATOMS_FILE = "atoms_evaluation_results.jsonl"
GOALS_FILE = "goals_evaluation_results.jsonl"

objects_path = os.path.join(RESULTS_FOLDER, OBJECTS_FILE)
atoms_path = os.path.join(RESULTS_FOLDER, ATOMS_FILE)
goals_path = os.path.join(RESULTS_FOLDER, GOALS_FILE)

# Load all results
def load_jsonl(path):
    results = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results

evaluation_results_objects_all = load_jsonl(objects_path)
evaluation_results_atoms_all = load_jsonl(atoms_path)
evaluation_results_goals_all = load_jsonl(goals_path)

def print_json(title, payload):
    print(title)
    print(json.dumps(payload, indent=2))

# Compute metrics
metrics_objects = calculate_metrics_objects(evaluation_results_objects_all)
print_json("Objects Evaluation Metrics:", metrics_objects)

metrics_atoms = calculate_metrics(evaluation_results_atoms_all)
print_json("Atoms Evaluation Metrics:", metrics_atoms)

metrics_goals = calculate_metrics(evaluation_results_goals_all)
print_json("Goals Evaluation Metrics:", metrics_goals)

# Compute metrics per domain
objects_per_domain = calculate_metrics_objects_per_domain(evaluation_results_objects_all)
print_json("Objects Metrics Per Domain:", objects_per_domain)

atoms_per_domain = calculate_metrics_per_domain(evaluation_results_atoms_all)
print_json("Atoms Metrics Per Domain:", atoms_per_domain)

goals_per_domain = calculate_metrics_per_domain(evaluation_results_goals_all)
print_json("Goals Metrics Per Domain:", goals_per_domain)

# Create subfolders for CSV and diagrams
# csv_folder = os.path.join(RESULTS_FOLDER, "csv")
# diag_folder = os.path.join(RESULTS_FOLDER, "diagrams")
# if not os.path.exists(csv_folder):
#     os.makedirs(csv_folder)
# if not os.path.exists(diag_folder):
#     os.makedirs(diag_folder)

# def results_to_csv_and_heatmap(results, csv_filename, heatmap_filename, value_metric="f1"):
#     """
#     Converts results to a CSV and generates a heatmap.
#     It assumes that each result dict contains keys:
#     - "request_id"
#     - "scenario"
#     - "precision"
#     - "recall"
#     - "f1"
#     and that 'batch_id' contains the domain name in the format: ..._domain_{domain}
#     """

#     if not results:
#         return

#     data = []
#     for r in results:
#         request_id = r.get("request_id", "")
#         scenario = r.get("scenario", "")
#         # Extract domain if possible
#         domain = request_id.split("_domain_")[-1] if "_domain_" in request_id else "unknown"
#         data.append({
#             "domain": domain,
#             "scenario": scenario,
#             "precision": r.get("precision", None),
#             "recall": r.get("recall", None),
#             "f1": r.get("f1", None)
#         })

#     df = pd.DataFrame(data)
#     df.to_csv(os.path.join(csv_folder, csv_filename), index=False)

#     if df.empty:
#         return

#     pivot = df.pivot_table(index="domain", columns="scenario", values=value_metric, aggfunc='mean')

#     plt.figure(figsize=(10,6))
#     sns.heatmap(pivot, annot=True, cmap="Blues", fmt=".2f")
#     plt.title(f"{csv_filename.replace('.csv','').title()} Mean {value_metric.upper()} by Domain and Scenario")
#     plt.tight_layout()
#     plt.savefig(os.path.join(diag_folder, heatmap_filename))
#     plt.close()

# Generate CSV and heatmaps for objects, atoms, and goals (F1)
# results_to_csv_and_heatmap(evaluation_results_objects_all, "objects_metrics.csv", "objects_heatmap_f1.png", value_metric="f1")
# results_to_csv_and_heatmap(evaluation_results_atoms_all, "atoms_metrics.csv", "atoms_heatmap_f1.png", value_metric="f1")
# results_to_csv_and_heatmap(evaluation_results_goals_all, "goals_metrics.csv", "goals_heatmap_f1.png", value_metric="f1")

# # Also create heatmaps for precision
# results_to_csv_and_heatmap(evaluation_results_objects_all, "objects_metrics.csv", "objects_heatmap_precision.png", value_metric="precision")
# results_to_csv_and_heatmap(evaluation_results_atoms_all, "atoms_metrics.csv", "atoms_heatmap_precision.png", value_metric="precision")
# results_to_csv_and_heatmap(evaluation_results_goals_all, "goals_metrics.csv", "goals_heatmap_precision.png", value_metric="precision")

# # Also create heatmaps for recall
# results_to_csv_and_heatmap(evaluation_results_objects_all, "objects_metrics.csv", "objects_heatmap_recall.png", value_metric="recall")
# results_to_csv_and_heatmap(evaluation_results_atoms_all, "atoms_metrics.csv", "atoms_heatmap_recall.png", value_metric="recall")
# results_to_csv_and_heatmap(evaluation_results_goals_all, "goals_metrics.csv", "goals_heatmap_recall.png", value_metric="recall")

print("Aggregation completed. CSV and diagrams (including precision, recall, and F1 heatmaps) saved.")
