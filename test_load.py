import pickle
import numpy as np
import os

# Define the filename of your model
MODEL_FILE = 'best_diabetes_pipeline.pkl'

try:
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: Model file not found at: {MODEL_FILE}")

    with open(MODEL_FILE, 'rb') as f:
        # Load the object
        loaded_model = pickle.load(f)

    print(f"--- Model Loading Check ---")
    print(f"Successfully loaded object.")
    print(f"Object type is: {type(loaded_model)}")

    # Test if it has the .predict method (this is the critical check!)
    if hasattr(loaded_model, 'predict'):
        print("RESULT: SUCCESS. The loaded object has a '.predict()' method.")

        # Test a dummy prediction (using 8 features)
        # Features: Preg, Glu, BP, ST, Ins, BMI, DPF, Age
        dummy_data = np.array([[1, 120, 70, 30, 0, 30.0, 0.5, 30]])

        # The model expects a 2D array, which is why we wrap it in a second set of brackets
        test_prediction = loaded_model.predict(dummy_data)[0]
        print(f"Test prediction result (0 = No, 1 = Yes): {test_prediction}")

    else:
        print("RESULT: FAILURE. The loaded object does NOT have a '.predict()' method.")
        print("This means you saved a dictionary, list, or other non-model object.")

except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}")