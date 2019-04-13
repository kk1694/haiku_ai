
from fastai.text import *
import fastai
print(fastai.__version__)

from custom_callbacks import *


# Settings

path = Path('data')

epochs_head = 3
lr_head = slice(1e-2)

epochs_unfreeze = 10
lr_unfreeze = slice(3e-3)

epochs_finetune = 40
lr_finetune = slice(1e-5)

fn = 'haikus.csv'

# Prepare Data
print('Prepare Data')

data_lm = TextLMDataBunch.from_csv(path, fn, bptt=32, 
                                   bs = 64,
                                   max_vocab=10000, valid_pct=0.02,
                                   include_eos=True, include_bos=True)


print(f'Training/validation data set: {len(data_lm.train_ds), len(data_lm.valid_ds)}')


callback_fns=[SampleWriter, SaveModelCallback, CSVLogger]


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5, pretrained=True,
                               callback_fns=callback_fns)


# Start Fitting
print('Start training!')

learn.fit_one_cycle(epochs_head, lr_head)

print('Finished Head')

learn.unfreeze()
learn.fit_one_cycle(epochs_unfreeze, lr_finetune)
learn.save('awd_first_phase')

print('Finished First Phase')

learn.callback_fns.append(ReduceLROnPlateauCallback)
learn.fit(epochs_finetune, lr_finetune)
learn.save('awd_second_phase')

print('Finished Script')



