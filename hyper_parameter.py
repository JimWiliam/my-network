import os

os.environ['TF_DATA'] = '/media/todrip/数据/data/common'

weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.1
num_classes = 10
reduction_ratio = 4

batch_size = 128
iteration = 391
# 128 * 391 ~ 50,000

test_iteration = 10

total_epochs = 100

base_dir = '/media/todrip/数据/experiment'
