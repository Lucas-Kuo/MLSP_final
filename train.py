from cct import create_cct_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import config

# for labeled images
def load_images(imagePath):
    # pass in the image directory and set the class names
    return image_dataset_from_directory(
        imagePath, shuffle=True, labels='inferred', class_names=config.classes,
        label_mode="categorical", batch_size=config.batch_size,
        image_size=(config.image_size, config.image_size))

model = create_cct_model()
train_dataset = load_images(config.training_path)

optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)

model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
            tfa.metrics.F1Score(config.num_classes, name="macro-F1-score")
        ],
    )

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    config.checkpoint_filepath,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)

history = model.fit(
    x=train_dataset,
    batch_size=config.batch_size,
    epochs=config.num_epochs,
    validation_split=0.1,
    callbacks=[checkpoint_callback],
    verbose=1)

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.savefig(config.result_pic_name)
plt.show()
