import os
import pandas as pd

## Hyperparameters and constants

positional_emb = True
conv_layers = 2
projection_dim = 128

num_heads = 2
transformer_units = [
    projection_dim,
    projection_dim,
    projection_dim
]
transformer_layers = 3
stochastic_depth_rate = 0.1

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 8
num_epochs = 30
image_size = 64

num_classes = 219
input_shape = (image_size, image_size, 3)

public_test_path = "./orchid_public_set"
private_test_path = "./orchid_private_set"
label_path = "./training/label.csv"
training_orig_path = "./training"
training_path = "./train"
checkpoint_filepath = "savedmodel/checkpoint"
result_pic_name = "training_curve.jpg"

df = pd.read_csv(label_path)
classes = [str(x) for x in df["category"].unique().tolist()]
