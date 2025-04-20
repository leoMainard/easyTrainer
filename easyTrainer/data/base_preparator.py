from abc import ABC, abstractmethod

from typing import Union, Tuple, Dict

class BasePreparator(ABC):
    """
    A base class for text preparators, defines the interface and common utilities.
    """
    def __init__(self):
        self.order = {}
        self.params: Dict[str, Union[Tuple, str, int, bool]] = {}

    def __repr__(self):
        return f"{self.__class__.__name__}(order={self.order})"

    def __str__(self):
        ordered = ', '.join(f"{name}@{order}" for name, order in self.order.items())
        return f"{self.__class__.__name__} with transformations: {ordered}"