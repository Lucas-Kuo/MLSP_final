from cct import create_cct_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import os
import pandas as pd
import config

def load_images(imagePath):
    # pass in the path of testing dataset
    # since we're making inferences, there will be no labels
    return image_dataset_from_directory(
        imagePath, labels=None, shuffle=False, label_mode=None,
        batch_size=config.batch_size, image_size=(config.image_size, config.image_size))

# load testing dataset (public)
test_dataset = load_images(config.public_test_path)

# construct model and load weights
model = create_cct_model()
model.load_weights(config.checkpoint_filepath)

# predictions for public test dataset
print("[INFO] making public inferences...")
predictions = model.predict(test_dataset)

# predictions are 200-dimentional vectors
# we only keep the largest ones' indices
output = list(np.argmax(predictions, axis=1))

# map the image file names to their respective result
result = []
N = len(output)
for i in range(N):
#     name = test_dataset.file_paths[i][-8:]
    name = test_dataset.file_paths[i].split(os.path.sep)[-1]
    label = config.classes[output[i]]
#     result[name] = label
    result.append([name, label])
    
# load testing dataset (private)
test_dataset = load_images(config.private_test_path)

# predictions for private test dataset
print("[INFO] making private inferences...")
predictions = model.predict(test_dataset)
output = list(np.argmax(predictions, axis=1))

N = len(output)
for i in range(N):
    name = test_dataset.file_paths[i].split(os.path.sep)[-1]
    label = config.classes[output[i]]
    result.append([name, label])

# create submission
print("[INFO] creating submission file...")
df = pd.DataFrame(data=result, columns=["filename", "category"])
df.to_csv("submission.csv")
