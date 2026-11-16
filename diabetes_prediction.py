'''
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle  # For saving the best model

# Import the new boosting algorithms
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')


class DiabetesPrediction:
    def __init__(self, mongo_uri, db_name, collection_name):
        """
        Initialize the Diabetes Prediction System
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.scaler = StandardScaler()
        self.models = {}
        self.ensemble_models = {}
        self.results = {}

    def connect_mongodb(self):
        """Connect to MongoDB and fetch data"""
        try:
            # Added serverSelectionTimeoutMS to prevent indefinite hang
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)

            # The ismaster command is cheap and does not require auth.
            client.admin.command('ping')

            db = client[self.db_name]
            collection = db[self.collection_name]

            # Fetch all documents
            data = list(collection.find())
            print(f"✓ Successfully connected to MongoDB at {self.mongo_uri}")
            print(f"✓ Retrieved {len(data)} records from {self.collection_name}")

            # Convert to DataFrame and remove MongoDB _id field
            df = pd.DataFrame(data)
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)

            client.close()
            return df

        except Exception as e:
            print(f"✗ CRITICAL ERROR: Could not connect or load data from MongoDB.")
            print(f"   Please check if your MongoDB server is running and the URI/DB/Collection are correct.")
            print(f"   Error details: {e}")
            return None

    def preprocess_data(self, df, target_column='Outcome'):
        print("\n--- Data Preprocessing ---")
        print(f"Dataset shape: {df.shape}")

        # Handle missing values if any
        df = df.fillna(df.median())

        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Scale features
        # Save the scaler instance (self.scaler is now fitted)
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train_balanced, y_test, X.columns

    def initialize_models(self):
        # Initializing all models, including the new boosting classifiers
        self.models = {
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(random_state=42),
            # --- NEW BOOSTING MODELS ---
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
            'CatBoost': CatBoostClassifier(verbose=0, random_state=42),  # verbose=0 suppresses training output
            'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)  # verbose=-1 suppresses warnings
        }

    def tune_hyperparameters(self, X_train, y_train):
        print("\n--- Hyperparameter Tuning ---")
        # Note: We only tune the original four models due to time complexity.

        svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1, 0.01], 'kernel': ['rbf', 'poly']}
        knn_params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'],
                      'metric': ['euclidean', 'manhattan']}
        rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30, None], 'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 4]}

        svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=5, scoring='f1', n_jobs=-1)
        svm_grid.fit(X_train, y_train)
        self.models['SVM'] = svm_grid.best_estimator_
        print(f"Best SVM params: {svm_grid.best_params_}")

        knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='f1', n_jobs=-1)
        knn_grid.fit(X_train, y_train)
        self.models['KNN'] = knn_grid.best_estimator_
        print(f"Best KNN params: {knn_grid.best_params_}")

        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='f1', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        self.models['Random Forest'] = rf_grid.best_estimator_
        print(f"Best RF params: {rf_grid.best_params_}")

    def train_individual_models(self, X_train, y_train):
        print("\n--- Training Individual Models (Including Boosting) ---")
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            print(f"{name} - CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    def create_ensemble_models(self, X_train, y_train):
        print("\n--- Creating Ensemble Models (Stacking Enhanced with Boosting) ---")

        # Base estimators for the Voting Classifier (only includes models with predict_proba for soft voting)
        voting_clf = VotingClassifier(
            estimators=[('svm', self.models['SVM']),
                        ('nb', self.models['Naive Bayes']),
                        ('rf', self.models['Random Forest']),
                        ('xgb', self.models['XGBoost']),
                        ('lgbm', self.models['LightGBM']),
                        ('cat', self.models['CatBoost'])],
            voting='soft'
        )

        # Stacking Classifier now includes all individual models for maximal feature representation
        stacking_clf = StackingClassifier(
            estimators=[
                ('svm', self.models['SVM']),
                ('knn', self.models['KNN']),
                ('nb', self.models['Naive Bayes']),
                ('rf', self.models['Random Forest']),
                ('xgb', self.models['XGBoost']),
                ('cat', self.models['CatBoost']),
                ('lgbm', self.models['LightGBM']),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )

        # Bagging Classifier uses the most stable single estimator (Random Forest)
        bagging_clf = BaggingClassifier(
            estimator=RandomForestClassifier(random_state=42),
            n_estimators=10,
            random_state=42
        )

        self.ensemble_models = {
            'Voting Classifier (Soft)': voting_clf,
            'Stacking Classifier (Enhanced)': stacking_clf,
            'Bagging Classifier': bagging_clf
        }

        for name, model in self.ensemble_models.items():
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            print(f"{name} - CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    def evaluate_models(self, X_test, y_test):
        print("\n--- Model Evaluation ---")
        all_models = {**self.models, **self.ensemble_models}
        for name, model in all_models.items():
            y_pred = model.predict(X_test)
            # Use try-except block for predict_proba in case a model doesn't support it (like some non-tuned Naive Bayes)
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except AttributeError:
                # Fallback for models without predict_proba (shouldn't happen here, but good practice)
                y_pred_proba = y_pred
                roc_auc = 'N/A'

            self.results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                'ROC-AUC': roc_auc,
                'Confusion Matrix': confusion_matrix(y_test, y_pred),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        self.print_results()

    def print_results(self):
        print("\n" + "=" * 105)
        print("PERFORMANCE METRICS (Including New Boosting Algorithms)")
        print(f"{'Model':<35} {'Accuracy':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("=" * 105)
        for name, metrics in self.results.items():
            auc_display = f"{metrics['ROC-AUC']:.4f}" if isinstance(metrics['ROC-AUC'], float) else metrics['ROC-AUC']
            print(f"{name:<35} {metrics['Accuracy']:<10.4f} {metrics['Precision']:<12.4f} "
                  f"{metrics['Recall']:<12.4f} {metrics['F1-Score']:<12.4f} {auc_display:<12}")
        print("=" * 105)
        best_f1 = max(self.results.items(),
                      key=lambda x: x[1]['F1-Score'] if isinstance(x[1]['F1-Score'], (int, float)) else -1)
        print(f"\n Best Model (by F1-Score): {best_f1[0]} ({best_f1[1]['F1-Score']:.4f})")

    # [Rest of the methods - plot_results, predict_new_sample - remain unchanged]
    # ... (Plotting code is kept to maintain original file functionality) ...
    def plot_results(self, y_test):
        # Placeholder to prevent error if run outside an environment that supports plt.show()
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 1. Metrics Comparison
            metrics_df = pd.DataFrame({
                name: [results['Accuracy'], results['Precision'],
                       results['Recall'], results['F1-Score'],
                       results['ROC-AUC'] if isinstance(results['ROC-AUC'], float) else 0]
                for name, results in self.results.items()
            }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])

            metrics_df.T.plot(kind='bar', ax=axes[0, 0], rot=45)
            axes[0, 0].set_title('Model Performance Metrics Comparison')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].legend(loc='lower right')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Confusion Matrix for Best Model (by F1-Score)
            best_model_name = max(self.results.items(),
                                  key=lambda x: x[1]['F1-Score'] if isinstance(x[1]['F1-Score'], (int, float)) else -1)[
                0]
            cm = self.results[best_model_name]['Confusion Matrix']

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                        xticklabels=['No Diabetes', 'Diabetes'],
                        yticklabels=['No Diabetes', 'Diabetes'])
            axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
            axes[0, 1].set_ylabel('True Label')
            axes[0, 1].set_xlabel('Predicted Label')

            # 3. ROC Curves
            for name, results in self.results.items():
                if isinstance(results['ROC-AUC'], float):
                    fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
                    axes[0, 2].plot(fpr, tpr, label=f"{name} (AUC={results['ROC-AUC']:.3f})")

            axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0, 2].set_xlabel('False Positive Rate')
            axes[0, 2].set_ylabel('True Positive Rate')
            axes[0, 2].set_title('ROC Curves Comparison')
            axes[0, 2].legend(loc='lower right', fontsize=8)
            axes[0, 2].grid(True, alpha=0.3)

            # 4. Accuracy Comparison
            accuracies = [results['Accuracy'] for results in self.results.values()]
            model_names = list(self.results.keys())

            bars = axes[1, 0].barh(model_names, accuracies, color='skyblue')
            axes[1, 0].set_xlabel('Accuracy')
            axes[1, 0].set_title('Model Accuracy Comparison')
            axes[1, 0].set_xlim([0.7, 1.0])

            # Add value labels on bars
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                axes[1, 0].text(acc, i, f' {acc:.4f}', va='center')

            axes[1, 0].grid(True, alpha=0.3, axis='x') 


            # 5. Recall vs Precision
            recalls = [results['Recall'] for results in self.results.values()]
            precisions = [results['Precision'] for results in self.results.values()]

            axes[1, 1].scatter(recalls, precisions, s=100, alpha=0.6, c=range(len(model_names)), cmap='viridis')
            for i, name in enumerate(model_names):
                axes[1, 1].annotate(name, (recalls[i], precisions[i]),
                                    fontsize=8, ha='right')

            axes[1, 1].set_xlabel('Recall')
            axes[1, 1].set_ylabel('Precision')
            axes[1, 1].set_title('Recall vs Precision')
            axes[1, 1].grid(True, alpha=0.3)

            # 6. F1-Score Comparison
            f1_scores = [results['F1-Score'] for results in self.results.values()]

            colors = ['#ff9999' if 'Ensemble' in name or 'Voting' in name or 'Stacking' in name or 'Bagging' in name
                      else '#66b3ff' for name in model_names]
            bars = axes[1, 2].bar(range(len(model_names)), f1_scores, color=colors)
            axes[1, 2].set_xticks(range(len(model_names)))
            axes[1, 2].set_xticklabels(model_names, rotation=45, ha='right')
            axes[1, 2].set_ylabel('F1-Score')
            axes[1, 2].set_title('F1-Score Comparison (Red=Ensemble)')
            axes[1, 2].grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, score in zip(bars, f1_scores):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width() / 2., height,
                                f'{score:.3f}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plots. Error: {e}")
    def save_best_model(self, filename='best_diabetes_pipeline.pkl'):  # Filename changed
        """Save the best performing model and the scaler."""

        # 1. Get the best model
        best_model_name = \
            max(self.results.items(),
                key=lambda x: x[1]['F1-Score'] if isinstance(x[1]['F1-Score'], (int, float)) else -1)[
                0]
        best_model = {**self.models, **self.ensemble_models}[best_model_name]

        # CRITICAL FIX: Save a tuple containing (Model, Scaler)
        pipeline_objects = (best_model, self.scaler)

        with open(filename, 'wb') as f:
            pickle.dump(pipeline_objects, f)

        print(f"\n✓ Best model ({best_model_name}) and Scaler saved to {filename}")

    def predict_new_sample(self, sample_data):
        """Use the best model to make a prediction on new data."""
        best_model_name = \
            max(self.results.items(),
                key=lambda x: x[1]['F1-Score'] if isinstance(x[1]['F1-Score'], (int, float)) else -1)[
                0]
        best_model = {**self.models, **self.ensemble_models}[best_model_name]

        if isinstance(sample_data, dict):
            sample_data = pd.DataFrame([sample_data])

        sample_scaled = self.scaler.transform(sample_data)
        prediction = best_model.predict(sample_scaled)
        probability = best_model.predict_proba(sample_scaled)

        result = {
            'best_model': best_model_name,
            'prediction': 'Diabetes' if prediction[0] == 1 else 'No Diabetes',
            'probability': {
                'No Diabetes': probability[0][0],
                'Diabetes': probability[0][1]
            }
        }
        return result


# Main execution
if __name__ == "__main__":
    #  STEP 1: REVIEW AND UPDATE THESE CONFIGURATIONS
    # =========================================================================
    MONGO_URI = "mongodb://localhost:27017/"  # <--- CONFIRM THIS URI IS CORRECT
    DB_NAME = "diabetics"  # <--- CONFIRM THIS DATABASE EXISTS
    COLLECTION_NAME = "sample"  # <--- CONFIRM THIS COLLECTION HAS DATA
    # =========================================================================

    # Initialize the system
    dp = DiabetesPrediction(MONGO_URI, DB_NAME, COLLECTION_NAME)

    print("=" * 70)
    print("DIABETES PREDICTION SYSTEM STARTING (Boosting Algorithms Added)...")
    print("=" * 70)

    # Step 1: Connect to MongoDB and load data
    df = dp.connect_mongodb()

    if df is not None and not df.empty:
        # The script only proceeds if data is loaded successfully
        print("\nDATA LOADED. BEGINNING ML PIPELINE...")

        # Step 2: Preprocess data
        X_train, X_test, y_train, y_test, feature_names = dp.preprocess_data(df, target_column='Outcome')

        # Steps 3-6: Initialize, Tune, Train, and Ensemble
        dp.initialize_models()
        dp.tune_hyperparameters(X_train, y_train)
        dp.train_individual_models(X_train, y_train)
        dp.create_ensemble_models(X_train, y_train)

        # Step 7 & 8: Evaluate and Visualize
        dp.evaluate_models(X_test, y_test)

        # NOTE: Plotting requires a GUI/display environment, which may fail in certain sandboxes.
        dp.plot_results(y_test)

        # Step 9: Save the best model
        # Saved to 'best_diabetes_pipeline.pkl'
        dp.save_best_model('best_diabetes_pipeline.pkl')

        # Step 10: Example prediction
        print("\n--- Example Prediction ---")
        sample = {'Pregnancies': 1, 'Glucose': 122, 'BloodPressure': 90, 'SkinThickness': 51, 'Insulin': 220,
                  'BMI': 49.7,
                  'DiabetesPedigreeFunction': 0.325, 'Age': 31}
        result = dp.predict_new_sample(sample)
        print(f"\nPrediction using {result['best_model']}: **{result['prediction']}**")
        print(f"Probability (Diabetes): {result['probability']['Diabetes']:.4f}")

        print("\n" + "=" * 80)
        print("PROJECT EXECUTION COMPLETED SUCCESSFULLY! PKL FILE CREATED.")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("PROJECT FAILED TO COMPLETE: MongoDB connection or data load failed.")
        print("Please resolve the connection error or ensure your collection is not empty.")
        print("=" * 80)
        '''

import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle  # For saving the best model

# Import the new boosting algorithms
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')


class DiabetesPrediction:
    def __init__(self, mongo_uri, db_name, collection_name):
        """
        Initialize the Diabetes Prediction System
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.scaler = StandardScaler()
        self.models = {}
        self.ensemble_models = {}
        self.results = {}

    def connect_mongodb(self):
        """Connect to MongoDB and fetch data"""
        try:
            # Added serverSelectionTimeoutMS to prevent indefinite hang
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)

            # The ismaster command is cheap and does not require auth.
            client.admin.command('ping')

            db = client[self.db_name]
            collection = db[self.collection_name]

            # Fetch all documents
            data = list(collection.find())
            print(f"✓ Successfully connected to MongoDB at {self.mongo_uri}")
            print(f"✓ Retrieved {len(data)} records from {self.collection_name}")

            # Convert to DataFrame and remove MongoDB _id field
            df = pd.DataFrame(data)
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)

            client.close()
            return df

        except Exception as e:
            print(f"✗ CRITICAL ERROR: Could not connect or load data from MongoDB.")
            print(f"   Please check if your MongoDB server is running and the URI/DB/Collection are correct.")
            print(f"   Error details: {e}")
            return None

    def preprocess_data(self, df, target_column='Outcome'):
        print("\n--- Data Preprocessing ---")
        print(f"Dataset shape: {df.shape}")

        # Handle missing values if any
        df = df.fillna(df.median())

        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Scale features
        # Save the scaler instance (self.scaler is now fitted)
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train_balanced, y_test, X.columns

    def initialize_models(self):
        # Initializing all models, including the new boosting classifiers
        self.models = {
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(random_state=42),
            # --- NEW BOOSTING MODELS ---
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
            'CatBoost': CatBoostClassifier(verbose=0, random_state=42),  # verbose=0 suppresses training output
            'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)  # verbose=-1 suppresses warnings
        }

    def tune_hyperparameters(self, X_train, y_train):
        print("\n--- Hyperparameter Tuning ---")
        # Note: We only tune the original four models due to time complexity.

        svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1, 0.01], 'kernel': ['rbf', 'poly']}
        knn_params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'],
                      'metric': ['euclidean', 'manhattan']}
        rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30, None], 'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 4]}

        svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=5, scoring='f1', n_jobs=-1)
        svm_grid.fit(X_train, y_train)
        self.models['SVM'] = svm_grid.best_estimator_
        print(f"Best SVM params: {svm_grid.best_params_}")

        knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='f1', n_jobs=-1)
        knn_grid.fit(X_train, y_train)
        self.models['KNN'] = knn_grid.best_estimator_
        print(f"Best KNN params: {knn_grid.best_params_}")

        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='f1', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        self.models['Random Forest'] = rf_grid.best_estimator_
        print(f"Best RF params: {rf_grid.best_params_}")

    def train_individual_models(self, X_train, y_train):
        print("\n--- Training Individual Models (Including Boosting) ---")
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            print(f"{name} - CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    def create_ensemble_models(self, X_train, y_train):
        print("\n--- Creating Ensemble Models (Stacking Enhanced with Boosting) ---")

        # Base estimators for the Voting Classifier (only includes models with predict_proba for soft voting)
        voting_clf = VotingClassifier(
            estimators=[('svm', self.models['SVM']),
                        ('nb', self.models['Naive Bayes']),
                        ('rf', self.models['Random Forest']),
                        ('xgb', self.models['XGBoost']),
                        ('lgbm', self.models['LightGBM']),
                        ('cat', self.models['CatBoost'])],
            voting='soft'
        )

        # Stacking Classifier now includes all individual models for maximal feature representation
        stacking_clf = StackingClassifier(
            estimators=[
                ('svm', self.models['SVM']),
                ('knn', self.models['KNN']),
                ('nb', self.models['Naive Bayes']),
                ('rf', self.models['Random Forest']),
                ('xgb', self.models['XGBoost']),
                ('cat', self.models['CatBoost']),
                ('lgbm', self.models['LightGBM']),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )

        # Bagging Classifier uses the most stable single estimator (Random Forest)
        bagging_clf = BaggingClassifier(
            estimator=RandomForestClassifier(random_state=42),
            n_estimators=10,
            random_state=42
        )

        self.ensemble_models = {
            'Voting Classifier (Soft)': voting_clf,
            'Stacking Classifier (Enhanced)': stacking_clf,
            'Bagging Classifier': bagging_clf
        }

        for name, model in self.ensemble_models.items():
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            print(f"{name} - CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    def evaluate_models(self, X_test, y_test):
        print("\n--- Model Evaluation ---")
        all_models = {**self.models, **self.ensemble_models}
        for name, model in all_models.items():
            y_pred = model.predict(X_test)
            # Use try-except block for predict_proba in case a model doesn't support it (like some non-tuned Naive Bayes)
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except AttributeError:
                # Fallback for models without predict_proba (shouldn't happen here, but good practice)
                y_pred_proba = y_pred
                roc_auc = 'N/A'

            self.results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                'ROC-AUC': roc_auc,
                'Confusion Matrix': confusion_matrix(y_test, y_pred),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        self.print_results()

    def print_results(self):
        print("\n" + "=" * 105)
        print("PERFORMANCE METRICS (Including New Boosting Algorithms)")
        print(f"{'Model':<35} {'Accuracy':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("=" * 105)
        for name, metrics in self.results.items():
            auc_display = f"{metrics['ROC-AUC']:.4f}" if isinstance(metrics['ROC-AUC'], float) else metrics['ROC-AUC']
            print(f"{name:<35} {metrics['Accuracy']:<10.4f} {metrics['Precision']:<12.4f} "
                  f"{metrics['Recall']:<12.4f} {metrics['F1-Score']:<12.4f} {auc_display:<12}")
        print("=" * 105)
        best_f1 = max(self.results.items(),
                      key=lambda x: x[1]['F1-Score'] if isinstance(x[1]['F1-Score'], (int, float)) else -1)
        print(f"\n Best Model (by F1-Score): {best_f1[0]} ({best_f1[1]['F1-Score']:.4f})")

    def plot_results(self, y_test):
        """Generates and displays the three most critical plots for the academic report."""
        try:
            # Create subplots for a single row of 3 plots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 1. Metrics Comparison (Left Plot)
            metrics_df = pd.DataFrame({
                name: [results['Accuracy'], results['Precision'],
                       results['Recall'], results['F1-Score'],
                       results['ROC-AUC'] if isinstance(results['ROC-AUC'], float) else 0]
                for name, results in self.results.items()
            }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])

            # axes[0] corresponds to the first plot in the single row
            metrics_df.T.plot(kind='bar', ax=axes[0], rot=45)
            axes[0].set_title('Model Performance Metrics Comparison')
            axes[0].set_ylabel('Score')
            axes[0].legend(loc='lower right', fontsize=8)
            axes[0].grid(True, alpha=0.3)

            # 2. Confusion Matrix for Best Model (Middle Plot)
            best_model_name = max(self.results.items(),
                                  key=lambda x: x[1]['F1-Score'] if isinstance(x[1]['F1-Score'], (int, float)) else -1)[
                0]
            cm = self.results[best_model_name]['Confusion Matrix']

            # axes[1] corresponds to the second plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                        xticklabels=['No Diabetes', 'Diabetes'],
                        yticklabels=['No Diabetes', 'Diabetes'])
            axes[1].set_title(f'Confusion Matrix - {best_model_name}')
            axes[1].set_ylabel('True Label')
            axes[1].set_xlabel('Predicted Label')

            # 3. ROC Curves (Right Plot)
            # axes[2] corresponds to the third plot
            for name, results in self.results.items():
                if isinstance(results['ROC-AUC'], float):
                    fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
                    axes[2].plot(fpr, tpr, label=f"{name} (AUC={results['ROC-AUC']:.3f})")

            axes[2].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[2].set_xlabel('False Positive Rate')
            axes[2].set_ylabel('True Positive Rate')
            axes[2].set_title('ROC Curves Comparison')
            axes[2].legend(loc='lower right', fontsize=8)
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Warning: Could not display plots. Error: {e}")

    def save_best_model(self, filename='best_diabetes_pipeline.pkl'):  # Filename changed
        """Save the best performing model and the scaler."""

        # 1. Get the best model
        best_model_name = \
            max(self.results.items(),
                key=lambda x: x[1]['F1-Score'] if isinstance(x[1]['F1-Score'], (int, float)) else -1)[
                0]
        best_model = {**self.models, **self.ensemble_models}[best_model_name]

        # CRITICAL FIX: Save a tuple containing (Model, Scaler)
        pipeline_objects = (best_model, self.scaler)

        with open(filename, 'wb') as f:
            pickle.dump(pipeline_objects, f)

        print(f"\n✓ Best model ({best_model_name}) and Scaler saved to {filename}")

    def predict_new_sample(self, sample_data):
        """Use the best model to make a prediction on new data."""
        best_model_name = \
            max(self.results.items(),
                key=lambda x: x[1]['F1-Score'] if isinstance(x[1]['F1-Score'], (int, float)) else -1)[
                0]
        best_model = {**self.models, **self.ensemble_models}[best_model_name]

        if isinstance(sample_data, dict):
            sample_data = pd.DataFrame([sample_data])

        sample_scaled = self.scaler.transform(sample_data)
        prediction = best_model.predict(sample_scaled)
        probability = best_model.predict_proba(sample_scaled)

        result = {
            'best_model': best_model_name,
            'prediction': 'Diabetes' if prediction[0] == 1 else 'No Diabetes',
            'probability': {
                'No Diabetes': probability[0][0],
                'Diabetes': probability[0][1]
            }
        }
        return result


# Main execution
if __name__ == "__main__":
    #  STEP 1: REVIEW AND UPDATE THESE CONFIGURATIONS
    # =========================================================================
    MONGO_URI = "mongodb://localhost:27017/"  # <--- CONFIRM THIS URI IS CORRECT
    DB_NAME = "diabetics"  # <--- CONFIRM THIS DATABASE EXISTS
    COLLECTION_NAME = "sample"  # <--- CONFIRM THIS COLLECTION HAS DATA
    # =========================================================================

    # Initialize the system
    dp = DiabetesPrediction(MONGO_URI, DB_NAME, COLLECTION_NAME)

    print("=" * 70)
    print("DIABETES PREDICTION SYSTEM STARTING (Boosting Algorithms Added)...")
    print("=" * 70)

    # Step 1: Connect to MongoDB and load data
    df = dp.connect_mongodb()

    if df is not None and not df.empty:
        # The script only proceeds if data is loaded successfully
        print("\nDATA LOADED. BEGINNING ML PIPELINE...")

        # Step 2: Preprocess data
        X_train, X_test, y_train, y_test, feature_names = dp.preprocess_data(df, target_column='Outcome')

        # Steps 3-6: Initialize, Tune, Train, and Ensemble
        dp.initialize_models()
        dp.tune_hyperparameters(X_train, y_train)
        dp.train_individual_models(X_train, y_train)
        dp.create_ensemble_models(X_train, y_train)

        # Step 7 & 8: Evaluate and Visualize
        dp.evaluate_models(X_test, y_test)

        # NOTE: Plotting requires a GUI/display environment, which may fail in certain sandboxes.
        dp.plot_results(y_test)

        # Step 9: Save the best model
        # Saved to 'best_diabetes_pipeline.pkl'
        dp.save_best_model('best_diabetes_pipeline.pkl')

        # Step 10: Example prediction
        print("\n--- Example Prediction ---")
        sample = {'Pregnancies': 1, 'Glucose': 122, 'BloodPressure': 90, 'SkinThickness': 51, 'Insulin': 220,
                  'BMI': 49.7,
                  'DiabetesPedigreeFunction': 0.325, 'Age': 31}
        result = dp.predict_new_sample(sample)
        print(f"\nPrediction using {result['best_model']}: **{result['prediction']}**")
        print(f"Probability (Diabetes): {result['probability']['Diabetes']:.4f}")

        print("\n" + "=" * 80)
        print("PROJECT EXECUTION COMPLETED SUCCESSFULLY! PKL FILE CREATED.")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("PROJECT FAILED TO COMPLETE: MongoDB connection or data load failed.")
        print("Please resolve the connection error or ensure your collection is not empty.")
        print("=" * 80)