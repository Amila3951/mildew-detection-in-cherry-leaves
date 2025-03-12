import joblib
import json
import os

def load_test_evaluation(version):
    try:
        evaluate_json_path = os.path.join("outputs", version, "evaluate.json")
        confusion_matrix_path = os.path.join("outputs", version, "confusion_matrix.joblib")

        with open(evaluate_json_path, "r") as f:
            evaluation = json.load(f)

        evaluation['confusion_matrix'] = joblib.load(confusion_matrix_path)
        return evaluation
    except FileNotFoundError:
        return None