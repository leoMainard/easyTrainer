
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score

from easyTrainer.models.base_model import BaseModel

class SklearnClassifierModel(BaseModel):
    """
    A wrapper class for scikit-learn classifiers to integrate training, testing,
    validation, and prediction with optional fallback logic.
    """
    def __init__(self, model):
        """
        Initializes the SklearnClassifierModel.

        Args:
            model (sklearn.base.BaseEstimator): The scikit-learn model to be used.
        """
        super().__init__(model=model)

    def fit(self, X, y, param_grid: dict = None, **kwargs):
        """
        Trains the model using GridSearchCV with the given parameter grid.

        Args:
            X (array-like):
                Training feature set.
            y (array-like):
                Training labels.
            param_grid (dict, optional):
                Dictionary with parameters names as keys and lists of 
                parameter settings to try as values. Defaults to empty dict.
            **kwargs:
                Additional keyword arguments passed to GridSearchCV.

        Returns:
            sklearn.base.BaseEstimator:
                The best fitted model.
        """
        if param_grid is None:
            param_grid = {}
        grid_search = GridSearchCV(self.model, param_grid, **kwargs)
        grid_search.fit(X, y)
        self.best_fit_model = grid_search.best_estimator_
        return self.best_fit_model

    def test(self, X, y):
        """
        Evaluates the model on the test set.

        Args:
            X (array-like):
                Test feature set.
            y (array-like):
                Test labels.

        Returns:
            pd.DataFrame:
                Classification report including precision, recall, and F1-score.
        """
        y_val_pred = self.best_fit_model.predict(X)
        return pd.DataFrame(classification_report(y, y_val_pred, output_dict=True))

    def validation(self, X, y, fallback_class: int, validation_thresholds: list = None):
        """
        Evaluates model performance on the validation set across different thresholds.

        Args:
            X_val (array-like):
                Validation feature set.
            y_val (array-like):
                Validation labels.
            fallback_class (int):
                Class to use when prediction confidence is below the threshold.
            validation_thresholds (list, optional):
                List of thresholds to test. 
                Defaults to np.arange(0.5, 1.0, 0.02).

        Returns:
            dict:
                Dictionary containing thresholds, accuracy, weighted/macro F1 scores, 
                and F1 scores per class.
        """

        results = {
            "thresholds": [],
            "accuracy": [],
            "f1_score_weighted": [],
            "f1_score_macro": [],
            "f1_per_classes": {label: [] for label in np.unique(y)},
        }

        if validation_thresholds is None:
            validation_thresholds = np.arange(0.5, 1.0, 0.02).tolist()

        for threshold in validation_thresholds:
            probabilities = self.best_fit_model.predict_proba(X)
            y_pred = []
            for prob in probabilities:
                max_prob = max(prob)
                if max_prob < threshold:
                    y_pred.append(fallback_class)
                else:
                    y_pred.append(self.best_fit_model.classes_[prob.argmax()])

            accuracy = accuracy_score(y, y_pred)
            f1_weighted = f1_score(y, y_pred, average="weighted")
            f1_macro = f1_score(y, y_pred, average="macro")
            f1_per_classes = f1_score(y, y_pred, average=None)

            results["thresholds"].append(threshold)
            results["accuracy"].append(accuracy)
            results["f1_score_weighted"].append(f1_weighted)
            results["f1_score_macro"].append(f1_macro)

            for idx, label in enumerate(np.unique(y)):
                results["f1_per_classes"][label].append(f1_per_classes[idx])

        return results
    
    def predict(self, X, threshold : float = None, fallback_class : int = None) -> list:
        """
        Predicts the class labels for the given data.

        If a threshold and fallback_class are provided, the model will predict the fallback_class
        when the highest probability is below the threshold. Otherwise, the model will predict the class
        with the highest probability.

        Args:
            X (array-like or pd.DataFrame):
                The input data to predict on.
            threshold (float, optional):
                The probability threshold below which the fallback_class will be predicted. 
                If None, the model will predict the class with the highest probability.
            fallback_class (int, optional): 
                The class to predict when the highest probability is below the threshold. 
                If None, this behavior is not applied.

        Returns:
            list: A list of predicted class labels.
        """
        if threshold is not None and fallback_class is not None:
            probabilities = self.best_fit_model.predict_proba(X)
            preds = []
            for prob in probabilities:
                max_prob = max(prob)
                if max_prob < threshold:
                    preds.append(fallback_class)
                else:
                    preds.append(self.best_fit_model.classes_[prob.argmax()])
                
            return preds
        return self.best_fit_model.predict(X)