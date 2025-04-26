from easytrainer.data.base_preparator import BasePreparator

class DummyPreparator(BasePreparator):
    """
    A dummy implementation of BasePreparator for testing purposes.
    """
    def __init__(self):
        super().__init__()
        self.order = {"dummy_transform": 1}

def test_base_preparator_initialization():
    preparator = BasePreparator()
    assert preparator.order == {}
    assert preparator.params == {}

def test_base_preparator_repr():
    preparator = BasePreparator()
    assert repr(preparator) == "BasePreparator(order={})"

def test_base_preparator_str():
    preparator = BasePreparator()
    assert str(preparator) == "BasePreparator with transformations: "

def test_dummy_preparator_order():
    dummy = DummyPreparator()
    assert dummy.order == {"dummy_transform": 1}

def test_dummy_preparator_repr():
    dummy = DummyPreparator()
    assert repr(dummy) == "DummyPreparator(order={'dummy_transform': 1})"

def test_dummy_preparator_str():
    dummy = DummyPreparator()
    assert str(dummy) == "DummyPreparator with transformations: dummy_transform@1"