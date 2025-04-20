import pytest
from easyTrainer.models.base_model import BaseModel
import os

def test_base_model_is_abstract():
    # Ensure BaseModel cannot be instantiated directly
    with pytest.raises(TypeError):
        BaseModel()

def test_base_model_methods_are_abstract():
    # Create a concrete subclass to test abstract methods
    class ConcreteModel(BaseModel):
        def fit(self, *args, **kwargs):
            pass

        def test(self, *args, **kwargs):
            pass

        def validation(self, *args, **kwargs):
            pass

        def predict(self, X):
            pass

    model = ConcreteModel()
    assert model is not None

def test_save_and_load():
    # Create a concrete subclass to test save and load methods
    class ConcreteModel(BaseModel):
        def fit(self, *args, **kwargs):
            pass

        def test(self, *args, **kwargs):
            pass

        def validation(self, *args, **kwargs):
            pass

        def predict(self, X):
            pass

    model = ConcreteModel()
    model.best_fit_model = {"key": "value"}  # Mock a model object

    # Save the model
    path = "test_model.pkl"
    model.save(path)

    # Load the model
    loaded_model = ConcreteModel.load(path)
    assert loaded_model.best_fit_model == {"key": "value"}

    # Clean up
    os.remove(path)