import config
import shutil
import os
import pandas as pd

if not os.path.exists(config.label_path):
    raise FileNotFoundError(f"{config.label_path} does not exist. Please make sure to unzip the trainging dataset first.")
df = pd.read_csv(config.label_path)

for row in df.itertuples(index=False):
    # row[0] is filename and row[1] is category
    target_dir = os.path.sep.join([config.training_path, str(row[1])])
    if not os.path.exists(target_dir):
        print("[INFO] creating '{}' directory".format(target_dir))
        os.makedirs(target_dir)
    orig_file_path = os.path.sep.join([config.training_orig_path, row[0]])
    new_file_path = os.path.sep.join([target_dir, row[0]])
    shutil.copy2(orig_file_path, new_file_path)
