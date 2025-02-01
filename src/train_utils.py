# train_utils.py
import time
import numpy as np
import tensorflow as tf
from data_utils import augment_image, preprocess_image

def python_generator(X_data, Y_data, augment_flag=True, shuffle=True):
    idxs = np.arange(len(X_data))
    if shuffle:
        np.random.shuffle(idxs)
    for i in idxs:
        img_255 = X_data[i]
        lbl = Y_data[i]
        if augment_flag:
            yield augment_image(img_255), lbl
        else:
            yield preprocess_image(img_255), lbl

def build_dataset(X_data, Y_data, batch_size=16, augment_flag=True, shuffle=True, repeat=False):
    """
    Same as before, but with an added `repeat` argument.
    If repeat=True, the dataset is repeated infinitely for multiple epochs.
    """
    def gen():
        yield from python_generator(X_data, Y_data, augment_flag, shuffle)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.int32),
        output_shapes=((224,224,3),())
    )
    # If repeat=True => repeat indefinitely
    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

class TimeToAccCallback(tf.keras.callbacks.Callback):
    """
    Callback to measure the time when validation accuracy
    first crosses each target in 'acc_targets'.
    We'll store results in a dictionary that you can read after training.
    """
    def __init__(self, acc_targets=[0.85, 0.90]):
        super().__init__()
        self.acc_targets = acc_targets
        self.start_time = None
        self.times_reached = {}  # target -> time
        for t in acc_targets:
            self.times_reached[t] = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # logs['val_accuracy'] is what we want
        val_acc = logs.get('val_accuracy')
        if val_acc is not None:
            for t in self.acc_targets:
                if self.times_reached[t] is None and val_acc >= t:
                    self.times_reached[t] = time.time() - self.start_time