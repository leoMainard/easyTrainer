# easytrainer

Prepare your data and train your models easily.

Works with scikit-learn.


## Installation
```python
pip install easytrainer
```

## Usage
### **Preparator**
---

Data preparators integrate preparation methods. When instantiating the preparator, simply choose the order of execution for each method and fill in the parameters if needed.

```python
from easytrainer.data.text import TextualPreparator

preparator = TextualPreparator(
    to_lower=1,
    drop_stopwords=2
)
```

`easytrainer.data.text.TextualPreparator`

| Method Name | Description | Type |
|:------------|:------------|:------|
| `to_lower`                       | Order in which to apply lowercase transformation      | `(Optional[int])`                               |
| `to_upper`                       | Order in which to apply uppercase transformation      | `(Optional[int])`                               |
| `drop_stopwords`                 | Order in which to remove French stopwords             | `(Optional[int])`                               |
| `drop_digits`                    | Order in which to remove digits                       | `(Optional[int])`                               |
| `lemmatize`                      | Order in which to apply lemmatization using spaCy     | `(Optional[int])`                               |
| `drop_special_characters`        | Order to remove special characters (punctuation etc.) | `(Optional[int])`                               |
| `drop_accents`                   | Order in which to remove accents from characters      | `(Optional[int])`                               |
| `drop_words_less_than_N_letters` | Either the order (int), or a tuple (order, n_letters, isalpha_flag) indicating the order, minimum number of letters, and whether to filter only alphabetic words | `(Optional[Union[int, Tuple[int, int, bool]]])` |

Coming soon:
`easytrainer.data.numeric`
`easytrainer.data.image`
`easytrainer.data.video`
`easytrainer.data.audio`

Then call ``prepare()`` to preprocess your data:

```python
data = [
    "J'adore cette musique, elle est magnifique.",
    "Je déteste ce film, il est ennuyeux."
]

results = preparator.prepare(data)
print(results["data"])
>>> ["j'adore musique, magnifique.", 'déteste film, ennuyeux.']
```

`TextualPreparator.prepare()`

| Parameters | Description | Type |
|:-----------|:------------|:-----|
| `data`        | Text data to preprocess. If a numpy array is provided, it will be converted to a Series (1D) or DataFrame (2D) | `(List[str] or pd.Series or pd.DataFrame or np.ndarray)` |
| `all`         | If True and data is a DataFrame, apply transformations to all columns (converted to str). If False, apply only to object/string columns | `bool` (default : False) |
| `encoder_name_to_fit` | If specified, encode the processed data using this method ('tfidf', 'word2vec') | `(Optional[str])` |
| `custom_encoder_to_fit` | A user-provided encoder with a `fit_transform()` method | `(Optional[object])`                 |
| `custom_encoder_fit` | A user-provided fitted encoder with a `transform()` method | `(Optional[object])`                 |

Returns a dictionary:
- If no encoder: a dict containing 'data' (the preprocessed data).
- If encoder is specified: a dict containing 'data', 'encoded_data', and 'vectorizer'.

<br>

### **Models**
---

Easily train and evaluate your models.

```python
from sklearn.ensemble import RandomForestClassifier

from easytrainer.models.classifier import SklearnClassifierModel

SCM = SklearnClassifierModel(
    model=RandomForestClassifier(n_estimators=100, random_state=42),
)

# Training
SCM.fit(X_train, y_train, param_grid={'n_estimators': [50, 100]}, cv=3)

# Testing
report = SCM.test(X_test, y_test)
print(report)

# Validation
validation_results = SCM.validation(X_val, y_val, fallback_class=0)

# Prediction
predictions = SCM.predict(X_test, threshold=0.7, fallback_class=0)
```

`easytrainer.models`


| Method Name | Description | Args | Returns |
|:------------|:------------|:-----|:--------|
| `fit` | Trains the model, with optional hyperparameter tuning using `GridSearchCV`. | `(X, y, param_grid=None, **kwargs)` | `self.best_fit_model` (fitted model) |
| `test` | Evaluates the model on a test set, returning a classification report. | `(X, y)` | `pd.DataFrame` containing precision, recall, f1-score, support per class |
| `evaluate` | Evaluates model performance across different probability thresholds, optionally using a fallback class if prediction confidence is too low. | `(X, y, fallback_class, validation_thresholds=None)` | `pd.DataFrame` with scores (accuracy, weighted/macro F1, per-threshold metrics) |
| `predict` | Makes predictions, optionally applying a threshold and fallback class. | `(X, threshold=None, fallback_class=None)` | `np.ndarray` (predicted labels) |
| `save` | Saves the fitted model to disk. | `(path)` | `None` |
| `load` | Loads a previously saved model from disk. | `(path)` | `SklearnClassifierModel` instance with loaded model |
