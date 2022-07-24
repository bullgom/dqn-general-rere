from abc import ABC, abstractmethod
from typing import Callable, Optional, Any
from torchvision.transforms import transforms as tf
from mytypes import Size, State
import torch.nn.functional as F
import torch

BATCH_DIM = "batch_dim"


class Preprocessing(ABC):

    @abstractmethod
    def output_size(self, size: Size) -> Size:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, state: State) -> State:
        raise NotImplementedError

    def reset(self) -> None:
        return


class AddBatchDim(Preprocessing):

    def __call__(self, state: State) -> State:
        state = state.unsqueeze(0)
        return state

    def output_size(self, size: Size) -> Size:
        size[BATCH_DIM] = 1
        return size


class MultiFrame(Preprocessing):

    def __init__(self, num_frames: int) -> None:
        super().__init__()
        self.frames: list[State] = []
        self.num_frames = num_frames

    def output_size(self, size: Size) -> Size:
        size["c"] = size["c"] * self.num_frames
        return size

    def __call__(self, state: State) -> State:
        if len(self.frames) == 0:
            # we don't use Tensor.repeat
            for _ in range(self.num_frames - 1):
                self.frames.append(state.clone().detach())

        if len(self.frames) >= self.num_frames:
            self.frames.pop(0)

        self.frames.append(state)

        new_state = torch.concat(self.frames, dim=1)
        return new_state

    def reset(self) -> None:
        self.frames.clear()


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

class Grayscale(WrappedPreprocessing):
    def __init__(self) -> None:
        super().__init__()
        self.inner = tf.Grayscale()

    def output_size(self, size: Size) -> Size:
        assert size["c"] == 3
        size["c"] = 1
        return size
