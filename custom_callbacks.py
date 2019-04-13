from time import time
from fastai.callbacks import *


class SampleWriter(LearnerCallback):
    "A `LearnerCallback` that writes samples at the end of every epoch for a language model."
    def __init__(self, learn:Learner,
                 start_str = 'xxbos', n_words = 20, n_times = 20,
                 folder: str = 'samples'): 
        super().__init__(learn)
        self.path = self.learn.path/folder
        self.path.mkdir(parents=True, exist_ok=True)
        self.epochs, self.duration = 0, 0
        self.start_str, self.n_words, self.n_times = start_str, n_words, n_times
        
    def get_fn(self):  
        "return a file that we'll write our samples to"
        n = len(self.path.ls())
        return open(self.path/('sample_'+str(n)+'.txt'), 'w')

    def on_train_begin(self, **kwargs: Any) -> None:
        
        self.start_time = time()

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        "Add a line with `epoch` number, `smooth_loss` and `last_metrics`."
        
        self.duration += time() - self.start_time
        self.start_time = time()
        
        val_loss = last_metrics[0] if last_metrics is not None else 'NA'
        
        with self.get_fn() as f:
            f.write(f'Epoch: {self.epochs}\n')
            f.write(f'Total training duration so far: {self.epochs}\n')
            f.write(f'Losses (train/valid): {smooth_loss.item(), val_loss}\n')
            f.write(f'\n\n')
            for i in range(self.n_times):
                f.write(f'Sample {i}\n\n')
                f.write(self.learn.predict(self.start_str, self.n_words, temperature=0.7))
                f.write(f'\n\n')
            f.close()
            
        self.epochs += 1