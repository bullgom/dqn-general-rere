import pytorch_lightning as pl

class TorchProfilerCallaback(pl.Callback):
    
    def __init__(self, profiler):
        super().__init__()
        self.profiler = profiler
    
    def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs) -> None:
        self.profiler.step()
        pl_module.log_dict(outputs)
    
