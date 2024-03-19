import tensorflow as tf
import numpy as np
from Model import build_unet
from Loss import dice_coef_loss
import os


EPOCHS = 20
N_AUGMENT = 4
BATCH_SIZE = 10
INPUT_SHAPE = (400, 400, 3)
VALIDATION_SPLIT = 0.8
LEARNING_RATE = 0.001
N_BATCHS = 90//BATCH_SIZE
LOSS_FUNCTION = dice_coef_loss
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


# Inputs Loading functions
def load_data(file_path):
    return np.load(file_path)


def load_data_from_npy_file(input_file_path, output_file_path):
    input_data = tf.numpy_function(load_data, [input_file_path], tf.float32)
    output_data = tf.numpy_function(load_data, [output_file_path], tf.float32)
    return input_data, output_data


# Model creation
print("Model Creation...")
unet_model = build_unet(INPUT_SHAPE)
unet_model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)

# List of inputs paths
print("Creating Database...")
num_files = 100  # TBD
input_folder = r"Inputs\train\input"
segmentation_folder = r"Inputs\train\segmentation"
input_file_paths = [os.path.join(input_folder, f"{i}.npy") for i in range(1, num_files+1)]
output_file_paths = [os.path.join(segmentation_folder, f"{i}.npy") for i in range(1, num_files+1)]

# Create dataset from .npy inputs files
input_dataset = tf.data.Dataset.from_tensor_slices(input_file_paths)
output_dataset = tf.data.Dataset.from_tensor_slices(output_file_paths)

dataset = tf.data.Dataset.zip((input_dataset, output_dataset)).map(load_data_from_npy_file)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.shuffle(buffer_size=num_files)

# Training
print("Training starts...")
history_list = []
for epoch in range(EPOCHS):
    print(f"Epoch..................{epoch}/{EPOCHS}")
    for input_batch, output_batch in dataset:
        history = unet_model.fit(x=input_batch, y=output_batch, validation_split=VALIDATION_SPLIT)
        history_list.append(history)
        unet_model.save(r"Models\UNETV3_model_chkpt" + str(len(history_list)) + ".h5")

unet_model.save(r"MODELS\UNET.h5")
