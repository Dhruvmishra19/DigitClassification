# test_model.py
# This script performs a check to see if the trained model file exists.

import os
import sys

def run_model_existence_test():
    """
    Checks if the 'best_model.joblib' file was created and saved.
    """
    model_path = "models/best_model.joblib"

    print(f"\n--- Running Test: Checking for file at '{model_path}' ---")

    if os.path.exists(model_path):
        print("✅ Test PASSED: Model file found.")
        sys.exit(0)  # Exit with a success code
    else:
        print("❌ Test FAILED: Model file not found.")
        sys.exit(1)  # Exit with a failure code

if __name__ == "__main__":
    run_model_existence_test()
