import torch
import logging

from utils.train_pipeline import Trainer, CheckPointArgs, TrainArgs
from utils.dataset import rbc_data

from models import *

experiment_name = 'rbc_data_16_4'
model_name = 'Multigrid'

BATCH_SIZE = 16

# 
training_args = TrainArgs(num_epochs = 100, batch_size = BATCH_SIZE, learning_rate = 1e-4)
checkpoint_args = CheckPointArgs(model_name, experiment_name)

# Set up dataset
input_length = 16
output_length = 4

data_prep = [torch.load('data/sample_0.pt'),
             torch.load('data/sample_1.pt'),
             torch.load('data/sample_2.pt'),
             torch.load('data/sample_4.pt')]

train_indices = list(range(3000))
valid_indices = list(range(3000, 4000))

train_ds = rbc_data(data_prep, train_indices, input_length, output_length, False)
valid_ds = rbc_data(data_prep, train_indices, input_length, output_length, False)

# set up model
model = MG((2, 64, 64), input_length, output_length)

logging_configs = {
    'filename' : f'{checkpoint_args.checkpoint_dir}/log/multigrid_log.log',
    'level'    : logging.INFO,
# 'format'   : "{asctime} {levelname:<8} {message}"
}

torch.cuda.empty_cache()

trainer = Trainer(model, train_ds, valid_ds, checkpoint_args, training_args, logging_config = logging_configs)
trainer.train()
