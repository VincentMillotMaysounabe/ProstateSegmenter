"""
Preprocessing pipeline
Execute Preprocessing.py to create Inputs datas from Dataset folder.
Dependant of LoadDataset.py
"""

# Creating pipeline to normalize & augment data
import tensorflow as tf
import numpy as np

# Resizing and rescaling pipeline
def resize_gen(input_shape: list[int]) -> tf.keras.layers:
    """
    Generates a resize keras layer to reshape images to targeted shape
    :param input_shape: list[int] targeted shape
    :return: keras resizing layer
    """
    return tf.keras.layers.Resizing(input_shape[0], input_shape[1])


def rescale_img(input_img : list) -> list:
    """
    Rescale an image to [-1,1] using keras rescaling function
    :param input_img: image to rescale (numpy array)
    :return: rescaled image (numpy array)
    """
    rescale = tf.keras.layers.Rescaling(2. / input_img.max(), offset=-1)
    return rescale(input_img)


# Augmentation pipeline (need the same seed for image and segmentation)
class Augment():
    """
    Object to Augment data with the same random seed
    Random augmentations are flip and rotations
    """
    def __init__(self):
        #super().__init__()
        self.seed = np.random.randint(0, 100)

    def process(self, inputs: list, labels: list) -> [list, list]:
        # Change the seed on each call so every image get a different transformation

        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.Sequential(
            [tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=self.seed),
             tf.keras.layers.RandomRotation(0.4, seed=self.seed)])
        self.augment_labels = tf.keras.Sequential(
            [tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=self.seed),
             tf.keras.layers.RandomRotation(0.4, seed=self.seed)])
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)

        return inputs, labels


# Actual pipeline for volume and segmentation
def DataPipeline(volume: list, segmentation: list, n_augment: int, input_shape: list[int]) -> [list, list]:
    """
    Normalize, resize and augment data
    :param volume: np.array : MRI volume
    :param segmentation: np.array : corresponding prostate segmentation volume
    :param n_augment: Total number of volume to be returned (>= 1)
    :param input_shape: Targeted shape
    :return: np.array, np.array : processed_volume, processed_segmentation
    """
    # Making it 3-dimensional
    volume_output = np.expand_dims(volume, axis=3)
    segmentation_output = np.expand_dims(segmentation, axis=3)
    resize = resize_gen(input_shape)
    # resize and rescale volume
    volume_reshape = [resize(rescale_img(volume_output[i, :, :])) for i in range(len(volume_output))]
    segmentation_reshape = [resize(segmentation_output[i, :, :]) for i in range(len(segmentation_output))]

    # data augmentation
    augmented = [volume_reshape]
    augmented_seg = segmentation_reshape
    for i in range(n_augment):
        # Instantiate Augment
        augmentation = Augment()
        augmented.append([])
        # Augment data and save results
        for j in range(len(volume_reshape)):
            result_vol, result_seg = augmentation.process(volume_reshape[j], segmentation_reshape[j])
            augmented[-1].append(result_vol)
            augmented_seg.append(result_seg)

    return augmented, augmented_seg

def preprocess(input_shape, train_split):
    from LoadDataset import ProstateXImageStorage

    dataset = ProstateXImageStorage(r'C:\Users\VMI\OneDrive - EDAP TMS\Bureau\TestIA')
    dataset.loadImages()

    BATCH_SIZE = 1  # Should stay at 1
    n_files = dataset.nbSubjects
    n_train_files = int(n_files*train_split)
    N_BATCHS = n_files // BATCH_SIZE

    for b in range(0, N_BATCHS):
        # inputs=[]
        segmentations = []
        k = BATCH_SIZE * b
        print(f"processing data {k}-{k+BATCH_SIZE}...")
        all_inputs = []
        for i in range(k, k + BATCH_SIZE):
            dataset.getData(i)
            seg = dataset.outputSeg
            vol = dataset.outputImg

            volumes, segmentations_or = DataPipeline(vol, seg, 5, input_shape )
            inputs = np.zeros((len(segmentations_or), input_shape[0], input_shape[1], input_shape[2]))
            print(f"Loading, prepocessing and augmenting MRI id n°{i}")

            for v in range(len(volumes)):
                for j in range(len(volumes[v])):
                    inputs[j + len(volumes[0]) * v, :, :, 1] = np.squeeze(volumes[v][j])
                    if j != 0:
                        inputs[j + len(volumes[0]) * v, :, :, 0] = np.squeeze(volumes[v][j - 1])
                    if j != (len(volumes[0]) - 1):
                        inputs[j + len(volumes[0]) * v, :, :, 2] = np.squeeze(volumes[v][j + 1])

                    segmentations.append(segmentations_or[j + len(volumes[0]) * v])

            if i != k:
                all_inputs = np.concatenate((all_inputs, inputs))
            else:
                all_inputs = inputs.copy()

        #Converting to float32
        all_inputs = np.array(all_inputs, dtype=np.float32)
        print("Saving preprocessed files...")
        for i in range(len(all_inputs)):
            if k <= n_train_files:
                np.save(fr'Inputs\train\input\{k+i}.npy', all_inputs[i])
                np.save(fr'Inputs\train\segmentation\{k+i}.npy', segmentations[i])
            else :
                np.save(fr'Inputs\test\input\{k+i}.npy', all_inputs[i])
                np.save(fr'Inputs\test\segmentation\{k+i}.npy', segmentations[i])

if __name__ == "__main__":
    INPUT_SHAPE = (400, 400, 3)
    preprocess(INPUT_SHAPE)