# adv_utils.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from data_utils import augment_image

def create_adversarial_pattern(model, x_in, y_in):
    with tf.GradientTape() as tape:
        tape.watch(x_in)
        preds = model(x_in, training=False)
        loss = SparseCategoricalCrossentropy(from_logits=True)(y_in, preds)
    grads = tape.gradient(loss, x_in)
    return tf.sign(grads)

def adv_python_generator(X_data, Y_data, model, epsilon=0.1, shuffle=True):
    idxs = np.arange(len(X_data))
    if shuffle:
        np.random.shuffle(idxs)
    for i in idxs:
        img_255 = X_data[i]
        lbl = Y_data[i]
        # do albumentations
        c_img = augment_image(img_255)
        c_tf  = tf.expand_dims(c_img, axis=0)
        l_tf  = tf.constant([lbl], dtype=tf.int32)

        sign_grad = create_adversarial_pattern(model, c_tf, l_tf)
        adv_img = tf.clip_by_value(c_tf + epsilon*sign_grad, 0.0, 1.0)
        yield tf.concat([c_tf, adv_img], axis=0), tf.concat([l_tf,l_tf], axis=0)

def build_adv_dataset(X_data, Y_data, model, epsilon=0.1, shuffle=True, batch_size=1):
    def gen():
        yield from adv_python_generator(X_data, Y_data, model, epsilon, shuffle)
    ds = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.int32),
        output_shapes=((2,224,224,3),(2,))
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def adv_train_loop(model, X_data, Y_data, epochs=1, epsilon=0.1):
    optimizer = model.optimizer
    for e in range(epochs):
        print(f"\nAdversarial Epoch {e+1}/{epochs} (epsilon={epsilon})")
        adv_ds = build_adv_dataset(X_data, Y_data, model, epsilon, True, 1)
        step_idx=0
        for combined_imgs, combined_lbls in adv_ds:
            combined_imgs = tf.reshape(combined_imgs, [-1,224,224,3])
            combined_lbls = tf.reshape(combined_lbls, [-1])
            with tf.GradientTape() as tape:
                logits = model(combined_imgs, training=True)
                loss = SparseCategoricalCrossentropy(from_logits=True)(combined_lbls, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            step_idx+=1
            if step_idx%100==0:
                print(f"  step={step_idx}, loss={loss.numpy():.4f}")