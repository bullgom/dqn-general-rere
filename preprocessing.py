from abc import ABC, abstractmethod
from typing import Callable, Optional, Any
from torchvision.transforms import transforms as tf
from mytypes import Size, State
import torch.nn.functional as F

BATCH_DIM = "batch_dim"

class Preprocessing(ABC):

    @abstractmethod
    def output_size(self, size: Size) -> Size:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

class AddBatchDim(Preprocessing):
    
    def __call__(self, state: State) -> State:
        state = state.unsqueeze(0)
        return state
    
    def output_size(self, size: Size) -> Size:
        size[BATCH_DIM] = 1
        return size
        

class WrappedPreprocessing(Preprocessing):
    def __init__(self) -> None:
        super().__init__()
        self.inner: Optional[Callable] = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.inner(*args, **kwds)


class ToTensor(WrappedPreprocessing):
    def __init__(self) -> None:
        super().__init__()
        self.inner = tf.ToTensor()

    def output_size(self, size: Size) -> Size:
        return size


class Resize(WrappedPreprocessing):
    def __init__(self, size: Size, *args, **kwargs) -> None:
        super().__init__()
        
        self.size = size
        tuple_size = (size["w"], size["h"])
        self.inner = tf.Resize(tuple_size, *args, **kwargs)

    def output_size(self, size: Size) -> Size:
        size.update(self.size)
        return size
