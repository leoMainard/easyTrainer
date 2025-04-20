from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model=None):
        self.model = model
        self.best_fit_model = None

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass

    @abstractmethod
    def validation(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def save(self, path):
        import joblib
        joblib.dump(self.best_fit_model, path)

    @classmethod
    def load(cls, path):
        import joblib
        instance = cls(model=None)
        instance.best_fit_model = joblib.load(path)
        return instance