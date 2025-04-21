import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from easyTrainer.models.classifier import SklearnClassifierModel

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    return X, y

@pytest.fixture
def classifier_model():
    model = RandomForestClassifier(random_state=42)
    return SklearnClassifierModel(model=model)

def test_fit_with_grid_search(classifier_model, sample_data):
    X, y = sample_data
    param_grid = {"n_estimators": [10, 50], "max_depth": [None, 5]}
    best_model = classifier_model.fit(X, y, param_grid=param_grid, cv=3)
    assert best_model is not None
    assert hasattr(best_model, "predict")

def test_test_method(classifier_model, sample_data):
    X, y = sample_data
    classifier_model.fit(X, y)
    report = classifier_model.test(X, y)
    assert isinstance(report, pd.DataFrame)
    assert "precision" in report.T.columns
    assert "recall" in report.T.columns
    assert "f1-score" in report.T.columns
    assert "support" in report.T.columns

def test_validation_method(classifier_model, sample_data):
    X, y = sample_data
    classifier_model.fit(X, y)
    results = classifier_model.validation(X, y, fallback_class=0, validation_thresholds=[0.5, 0.7])
    assert "thresholds" in results
    assert len(results["thresholds"]) == 2
    assert all(isinstance(score, float) for score in results["accuracy"])

def test_predict_with_threshold(classifier_model, sample_data):
    X, y = sample_data
    classifier_model.fit(X, y)
    predictions = classifier_model.predict(X, threshold=0.6, fallback_class=0)
    assert len(predictions) == len(y)
    assert all(isinstance(pred, np.int64) for pred in predictions)

def test_predict_without_threshold(classifier_model, sample_data):
    X, y = sample_data
    classifier_model.fit(X, y)
    predictions = classifier_model.predict(X)
    assert len(predictions) == len(y)
    assert all(isinstance(pred, np.int64) for pred in predictions)