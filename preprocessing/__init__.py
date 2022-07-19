from abc import ABC, abstractmethod
from typing import Callable, Optional, Any
from torchvision.transforms import transforms as tf


class Preprocessing(ABC):

    @abstractmethod
    def output_size(self, size: tuple[int]) -> tuple[int]:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


class WrappedPreprocessing(Preprocessing):
    def __init__(self) -> None:
        super().__init__()
        self.inner: Optional[Callable] = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


class ToTensor(WrappedPreprocessing):
    def __init__(self) -> None:
        super().__init__()
        self.inner = tf.ToTensor()

    def output_size(self, size: tuple[int]) -> tuple[int]:
        w, h, c = size
        return c, h, w

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.inner(*args, **kwds)
