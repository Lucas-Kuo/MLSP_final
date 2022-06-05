import os

## Hyperparameters and constants

positional_emb = True
conv_layers = 2
projection_dim = 128

num_heads = 2
transformer_units = [
    projection_dim,
    projection_dim,
]
transformer_layers = 2
stochastic_depth_rate = 0.1

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 30
image_size = 256

num_classes = 219
input_shape = (256, 256, 3)

public_test_path = "./orchid_public_set"
private_test_path = "./orchid_private_set"
