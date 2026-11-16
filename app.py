from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np
import os
import sys
from datetime import datetime
from pymongo import MongoClient
# --- THE FIX IS HERE ---
# Import the standard Python exceptions instead of old pymongo-specific ones
from socket import timeout as SocketTimeoutError
from pymongo.errors import PyMongoError  # Use PyMongoError for general MongoDB errors

# --- Configuration ---
app = Flask(__name__)

# TODO: IMPORTANT: REPLACE THIS WITH YOUR ACTUAL MONGODB CONNECTION STRING
# Example for a local MongoDB: "mongodb://localhost:27017/"
# Example for a MongoDB Atlas cluster: "mongodb+srv://user:password@cluster.mongodb.net/..."
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "diabetes_predictions"
MONGO_COLLECTION_NAME = "prediction_history"

# The required features for the model, in the exact order the model expects them.
FEATURE_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

MODEL_FILE = 'best_diabetes_pipeline.pkl'
MODEL = None
SCALER = None
MONGO_CLIENT = None

# --- Model Loading and Initialization ---

# Check if the model file exists and load it on application startup
try:
    if not os.path.exists(MODEL_FILE):
        print(f"CRITICAL ERROR: Model file not found at: {MODEL_FILE}", file=sys.stderr)
        # sys.exit(1) # We can comment this out to allow the app to run even without the model file for now

    with open(MODEL_FILE, 'rb') as f:
        loaded_objects = pickle.load(f)

    # Correctly unpacks the model and the scaler from the tuple/list/dict
    if isinstance(loaded_objects, (tuple, list)) and len(loaded_objects) == 2 and hasattr(loaded_objects[0], 'predict'):
        MODEL, SCALER = loaded_objects
        print(
            f"Successfully loaded Model ({type(MODEL).__name__}) and Scaler ({type(SCALER).__name__}) from {MODEL_FILE}")
    elif hasattr(loaded_objects, 'predict'):
        MODEL = loaded_objects
        SCALER = None
        print(f"WARNING: Loaded only Model ({type(MODEL).__name__}). Assuming model was trained on unscaled data.")
    else:
        raise ValueError(
            f"The loaded object ({type(loaded_objects).__name__}) is not a valid model or pipeline tuple/list.")

except Exception as e:
    print(f"CRITICAL ERROR loading pipeline from {MODEL_FILE}: {e}", file=sys.stderr)
    MODEL = None
    SCALER = None

# --- MongoDB Initialization ---
try:
    # Use SocketTimeoutError for the serverSelectionTimeoutMS check
    MONGO_CLIENT = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # The ismaster command is a lightweight way to check the connection
    MONGO_CLIENT.admin.command('ismaster')
    print("Successfully connected to MongoDB!")

    # Get the database and collection instances (these lines are not strictly needed here, but kept for context)
    # db = MONGO_CLIENT[MONGO_DB_NAME]
    # collection = db[MONGO_COLLECTION_NAME]

except (SocketTimeoutError, PyMongoError) as e:
    # Catch both connection timeout and general PyMongo operation errors
    print(f"WARNING: Could not connect to MongoDB at {MONGO_URI}. History saving disabled. Error: {e}", file=sys.stderr)
    MONGO_CLIENT = None


# --- Helper Functions ---

def validate_and_process_data(form_data):
    """
    Validates form data, converts inputs to a NumPy array, and SCALES the data
    if a scaler was successfully loaded.
    """
    input_values = []

    # 1. Collect and Validate Inputs
    for feature in FEATURE_NAMES:
        value_str = form_data.get(feature)

        if not value_str:
            return None, f"Missing input for {feature.replace('_', ' ')}"

        try:
            value = float(value_str)
            # Basic validation for non-negative values
            if value < 0:
                return None, f"{feature.replace('_', ' ')} cannot be negative."
            input_values.append(value)
        except ValueError:
            return None, f"Invalid numeric input for {feature.replace('_', ' ')}"

    # 2. Convert to NumPy array
    features_array = np.array([input_values])

    # Apply scaling using the SCALER object before returning the array
    if SCALER is not None:
        features_array_scaled = SCALER.transform(features_array)
        return features_array_scaled, None
    else:
        return features_array, None


def save_prediction_to_db(input_data, prediction_text, probability_formatted):
    """Saves the input data and prediction result to MongoDB."""
    if MONGO_CLIENT is None:
        return

    try:
        db = MONGO_CLIENT[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]

        # Combine all data into a single document
        document = {
            "timestamp": datetime.now(),
            "input_data": {k: float(v) if k in FEATURE_NAMES and v else None for k, v in input_data.items()},
            "prediction": prediction_text,
            "probability": probability_formatted
        }

        # Insert the document
        collection.insert_one(document)
        print(f"Successfully saved prediction to MongoDB: {document['prediction']}")

    except Exception as e:
        print(f"ERROR saving to MongoDB: {e}", file=sys.stderr)


# --- Routes ---

@app.route('/', methods=['GET'])
def home():
    """Renders the main input form on initial load."""

    if MODEL is None:
        return render_template('index.html',
                               features=FEATURE_NAMES,
                               input_data={},
                               error=f"CRITICAL: The machine learning model could not be loaded from '{MODEL_FILE}'.")

    return render_template('index.html', features=FEATURE_NAMES, input_data={})


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form."""
    if MODEL is None:
        return redirect(url_for('home'))

    input_data = {k: v for k, v in request.form.items()}

    # 1. Validate and process input (returns scaled array if SCALER is present)
    features_array, error_message = validate_and_process_data(request.form)

    if error_message:
        return render_template('index.html',
                               features=FEATURE_NAMES,
                               input_data=input_data,
                               error=error_message)

    try:
        # 2. Make Prediction
        prediction_result = MODEL.predict(features_array)[0]

        # 3. Get Probability
        probability_result = MODEL.predict_proba(features_array)[0][1]

        # 4. Format Results
        prediction_text = "DIABETES DETECTED" if prediction_result == 1 else "NO DIABETES"
        probability_formatted = f"{probability_result * 100:.2f}%"

        # 5. Save the result to MongoDB (New Step!)
        save_prediction_to_db(input_data, prediction_text, probability_formatted)

        # 6. Render result back to the template
        return render_template('index.html',
                               features=FEATURE_NAMES,
                               input_data=input_data,
                               prediction_result=prediction_text,
                               probability=probability_formatted)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template('index.html',
                               features=FEATURE_NAMES,
                               input_data=input_data,
                               error=f"An unexpected error occurred during prediction: {e}.")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)