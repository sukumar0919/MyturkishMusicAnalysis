# My_module.py

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

class MusicModelProject4:
    def __init__(self, dataset_path, target_column, model_algorithm=None,
                 scoring_metric='accuracy', additional_questions=None):
        """
        The MusicModel for Project 4.

        Parameters:
        - dataset_path (str): Path to the dataset CSV file.
        - target_column (str): Name of the target column.
        - model_algorithm: The scikit-learn model class (e.g., RandomForestClassifier, LogisticRegression).
        - scoring_metric (str): The evaluation metric for cross-validation.
        - additional_questions (list): List of additional research questions to be answered by the model.
        """
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.model_algorithm = model_algorithm
        self.scoring_metric = scoring_metric
        self.additional_questions = additional_questions

    def build_model(self):
        """
        Build and evaluate the scikit-learn model using cross-validation.

        Returns:
        - str: A string containing the cross-validation results.
        """
        # Load the dataset
        data = pd.read_csv(self.dataset_path)

        # Features (excluding the target column)
        X = data.drop(columns=[self.target_column])

        # Target variable
        y = data[self.target_column]

        # Create a pipeline with StandardScaler and the specified model algorithm
        model_pipeline = make_pipeline(StandardScaler(), self.model_algorithm)

        # Use StratifiedKFold for cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Initialize lists to store metrics for each fold
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        confusion_matrices = []

        # Perform cross-validation
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model_pipeline.fit(X_train, y_train)
            y_pred = model_pipeline.predict(X_test)

            # Calculate metrics for each fold
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
            recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
            confusion_matrices.append(confusion_matrix(y_test, y_pred))

        # Display the cross-validation results
        result_str = f"Cross-Validation Results:\n" \
                     f"Accuracy: {sum(accuracy_scores) / len(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})\n" \
                     f"Precision: {sum(precision_scores) / len(precision_scores):.4f}\n" \
                     f"Recall: {sum(recall_scores) / len(recall_scores):.4f}\n" \
                     f"Confusion Matrix (average):\n{np.mean(confusion_matrices, axis=0)}"
        print(result_str)  # Print the results in the notebook

        return result_str

if __name__ == "__main__":
    # Example Usage
    dataset_path = "C:/Users/emman/OneDrive/Desktop/MyMusicAnalysis/Acoustic Features.csv"
    target_column = 'Class'
    model_algorithm = RandomForestClassifier(random_state=42)  # You can change the model algorithm here
    scoring_metric = 'accuracy'  # You can change the scoring metric here
    additional_questions = ["What is the impact of feature X on the target variable?"]

    # Creating an instance of the MusicModelProject4 class
    model = MusicModelProject4(dataset_path, target_column, model_algorithm, scoring_metric, additional_questions)

    # Building and evaluating the model
    result = model.build_model()

