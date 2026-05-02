import os

# dataset
dataset_root_dir = './dataset'
if not os.path.exists(dataset_root_dir):
    os.makedirs(dataset_root_dir)

# training hyperparameters
epoch_num = 20
batch_size = 16
learning_rate = 5e-3
momentum = 0.9