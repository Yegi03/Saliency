# # # #!/usr/bin/env python
# # # # -*- coding: utf-8 -*-
# # #
# # # """
# # # main.py: Step-by-step approach. Each step can be run individually,
# # # so you can see exactly where something might fail or see partial results.
# # # """
# # #
# # # import warnings
# # # warnings.simplefilter(action='ignore', category=FutureWarning)
# # #
# # # import os
# # # import sys
# # # import time
# # # import numpy as np
# # # import pandas as pd
# # # import cv2
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # #
# # # from sklearn.model_selection import train_test_split
# # #
# # # # Import your custom modules
# # # from model_defs import build_functional_cnn, build_functional_vgg16
# # # from train_utils import (
# # #     augment_image, preprocess_image, python_generator,
# # #     build_dataset_via_generator, build_test_dataset_via_generator
# # # )
# # # from xai_utils import final_results_and_saliency
# # #
# # # print("All libraries imported successfully!")
# # #
# # # # ======  CONFIG  ======
# # # BASE_PATH = "data/COVID-19_Radiography_Dataset"
# # # subfolders = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
# # # NUM_CLASSES = 4
# # # TARGET_IMG_SIZE = (224, 224)
# # # TEST_RATIO = 0.15
# # # VAL_RATIO_OF_TRAIN = 0.1765
# # # BATCH_SIZE = 16
# # # EPOCHS = 1  # set 3 if you want more thorough
# # #
# # # def step_pause(msg):
# # #     """A small helper to pause execution, so you can check logs
# # #        or skip steps if you want."""
# # #     input(f"\n--- STEP: {msg} ---\nPress ENTER to continue... ")
# # #
# # # def main():
# # #     # STEP 1: Collect file paths
# # #     step_pause("STEP 1: Collecting file paths from subfolders")
# # #     all_data = []
# # #     label_map = {}
# # #
# # #     if not os.path.isdir(BASE_PATH):
# # #         sys.exit(f"Dataset folder not found at {BASE_PATH}")
# # #
# # #     for idx, folder_name in enumerate(subfolders):
# # #         label_map[folder_name] = idx
# # #         folder_path = os.path.join(BASE_PATH, folder_name, "images")
# # #         if not os.path.isdir(folder_path):
# # #             print(f"WARNING: Directory not found => {folder_path}. Skipping.")
# # #             continue
# # #         file_list = os.listdir(folder_path)
# # #         count_images = 0
# # #         for f in file_list:
# # #             if f.lower().endswith(('.jpg','.jpeg','.png')):
# # #                 all_data.append((os.path.join(folder_path, f), idx))
# # #                 count_images += 1
# # #         print(f"{folder_name}: {count_images} recognized images")
# # #
# # #     df = pd.DataFrame(all_data, columns=['filepath','label'])
# # #     print(f"Total images: {len(df)}")
# # #     print(df.head())
# # #
# # #     step_pause("STEP 1 done. Check above logs for recognized images")
# # #
# # #     # STEP 2: Load images with OpenCV
# # #     step_pause("STEP 2: Load images with OpenCV, building X, Y arrays")
# # #     X_list, Y_list = [], []
# # #     for i in range(len(df)):
# # #         fpath, lbl = df.iloc[i]['filepath'], df.iloc[i]['label']
# # #         img_bgr = cv2.imread(fpath)
# # #         if img_bgr is None:
# # #             print("Could not load:", fpath)
# # #             continue
# # #         img_bgr = cv2.resize(img_bgr, TARGET_IMG_SIZE)
# # #         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# # #         X_list.append(img_rgb)
# # #         Y_list.append(lbl)
# # #     X = np.array(X_list)
# # #     Y = np.array(Y_list)
# # #     print("X shape:", X.shape, "| Y shape:", Y.shape)
# # #
# # #     step_pause("STEP 2 done. Check shape of X, Y")
# # #
# # #     # STEP 3: Train/Val/Test split
# # #     step_pause("STEP 3: Splitting data into train/val/test")
# # #     X_trainval, X_test, Y_trainval, Y_test = train_test_split(
# # #         X, Y, test_size=TEST_RATIO, stratify=Y, random_state=42
# # #     )
# # #     X_train, X_val, Y_train, Y_val = train_test_split(
# # #         X_trainval, Y_trainval,
# # #         test_size=VAL_RATIO_OF_TRAIN, stratify=Y_trainval, random_state=42
# # #     )
# # #     print("Train shape:", X_train.shape, Y_train.shape)
# # #     print("Val shape:  ", X_val.shape,   Y_val.shape)
# # #     print("Test shape: ", X_test.shape,  Y_test.shape)
# # #
# # #     step_pause("STEP 3 done. Check shapes for train/val/test")
# # #
# # #     # STEP 4: Build models
# # #     step_pause("STEP 4: Building CNN & VGG16 models")
# # #     cnn_model = build_functional_cnn(input_shape=(224,224,3), num_classes=NUM_CLASSES)
# # #     vgg16_model = build_functional_vgg16(input_shape=(224,224,3), num_classes=NUM_CLASSES)
# # #
# # #     print("CNN & VGG16 built. Summaries:")
# # #     print(cnn_model.summary())
# # #     print(vgg16_model.summary())
# # #
# # #     step_pause("STEP 4 done. Models built & summarized")
# # #
# # #     # STEP 5: Train CNN
# # #     step_pause("STEP 5: Training CNN (no adv) for EPOCHS=" + str(EPOCHS))
# # #     train_ds_cnn_noadv = build_dataset_via_generator(X_train, Y_train, BATCH_SIZE, True, True)
# # #     val_ds_cnn_noadv   = build_dataset_via_generator(X_val,   Y_val,   BATCH_SIZE, False, False)
# # #     import time
# # #     start_time = time.time()
# # #     cnn_model.fit(train_ds_cnn_noadv, epochs=EPOCHS, validation_data=val_ds_cnn_noadv)
# # #     time_cnn_noadv = time.time() - start_time
# # #
# # #     step_pause("STEP 5 done. CNN trained. Check logs for accuracy/loss")
# # #
# # #     # STEP 6: Train VGG16
# # #     step_pause("STEP 6: Training VGG16 (no adv) for EPOCHS=" + str(EPOCHS))
# # #     train_ds_vgg_noadv = build_dataset_via_generator(X_train, Y_train, BATCH_SIZE, True, True)
# # #     val_ds_vgg_noadv   = build_dataset_via_generator(X_val,   Y_val,   BATCH_SIZE, False, False)
# # #     start_time = time.time()
# # #     vgg16_model.fit(train_ds_vgg_noadv, epochs=EPOCHS, validation_data=val_ds_vgg_noadv)
# # #     time_vgg_noadv = time.time() - start_time
# # #
# # #     step_pause("STEP 6 done. VGG16 trained. Check logs for accuracy/loss")
# # #
# # #     # STEP 7: Evaluate on test set
# # #     step_pause("STEP 7: Evaluating on test set => CNN & VGG16")
# # #     test_ds = build_test_dataset_via_generator(X_test, Y_test, batch_size=BATCH_SIZE)
# # #     cnn_loss, cnn_acc = cnn_model.evaluate(test_ds, verbose=0)
# # #     vgg_loss, vgg_acc = vgg16_model.evaluate(test_ds, verbose=0)
# # #     print(f"\nCNN(no adv) => Test Acc={cnn_acc:.4f}")
# # #     print(f"VGG16(no adv) => Test Acc={vgg_acc:.4f}")
# # #
# # #     step_pause("STEP 7 done. Check final test accuracies above")
# # #
# # #     # STEP 8: Prepare placeholders for adv or confidence drop
# # #     step_pause("STEP 8: Summaries & Saliency - set adv placeholders")
# # #     time_cnn_adv   = 0.0
# # #     time_vgg_adv   = 0.0
# # #     acc_cnn_adv    = 0.0
# # #     acc_vgg_adv    = 0.0
# # #     confdrop_cnn_noadv = 0.0
# # #     confdrop_cnn_adv   = 0.0
# # #     confdrop_vgg_noadv = 0.0
# # #     confdrop_vgg_adv   = 0.0
# # #     cnn_adv_model  = None
# # #     vgg16_adv_model= None
# # #
# # #     # STEP 9: Call final_results_and_saliency
# # #     step_pause("STEP 9: Generating final results table, bar plots, & saliency overlay")
# # #     final_results_and_saliency(
# # #         time_cnn_noadv, time_cnn_adv,
# # #         time_vgg_noadv, time_vgg_adv,
# # #         cnn_acc, acc_cnn_adv,
# # #         vgg_acc, acc_vgg_adv,
# # #         confdrop_cnn_noadv, confdrop_cnn_adv,
# # #         confdrop_vgg_noadv, confdrop_vgg_adv,
# # #         cnn_model, cnn_adv_model,
# # #         vgg16_model, vgg16_adv_model,
# # #         X_test, Y_test
# # #     )
# # #
# # #     print("\nAll steps done. Check the saved images: accuracy_bar.png, confdrop_bar.png, saliency_overlay.png")
# # #
# # # # ---- End of main function ----
# # #
# # # if __name__ == "__main__":
# # #     main()
# #
# #
# #
# # #!/usr/bin/env python
# # # -*- coding: utf-8 -*-
# #
# # """
# # A lean main.py that trains CNN & VGG16 quickly by sampling only a small subset
# # of the dataset (e.g. 50 images per class) and running 1 epoch.
# # Once you confirm it works, you can remove the sampling limit or increase epochs.
# # """
# #
# # import warnings
# # warnings.simplefilter(action='ignore', category=FutureWarning)
# #
# # import os
# # import time
# # import numpy as np
# # import pandas as pd
# # import cv2
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# #
# # from sklearn.model_selection import train_test_split
# #
# # # Import your custom modules
# # from model_defs import build_functional_cnn, build_functional_vgg16
# # from train_utils import (
# #     build_dataset_via_generator,
# #     build_test_dataset_via_generator
# # )
# # from xai_utils import final_results_and_saliency
# #
# # print("All libraries imported successfully!")
# #
# # # ------------------------------ CONFIG ------------------------------
# # BASE_PATH = "data/COVID-19_Radiography_Dataset"
# # subfolders = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
# #
# # NUM_CLASSES = 4
# # TARGET_IMG_SIZE = (224, 224)
# # TEST_RATIO = 0.15
# # VAL_RATIO_OF_TRAIN = 0.1765
# #
# # BATCH_SIZE = 16
# # EPOCHS = 1  # Quick test: only 1 epoch
# # MAX_IMAGES_PER_CLASS = 50  # Load only 50 per class (speedy)
# #
# # def main():
# #     """
# #     Lean pipeline:
# #     1) Gather file paths (limited to N per class).
# #     2) Load images (OpenCV).
# #     3) Split into train/val/test.
# #     4) Train CNN & VGG16 quickly (1 epoch).
# #     5) Evaluate test set.
# #     6) Show final results & saliency (with placeholders for adv models).
# #     """
# #     # 1) Collect file paths
# #     print(f"\n=== Step 1: Collecting up to {MAX_IMAGES_PER_CLASS} images per class ===")
# #     all_data = []
# #     label_map = {}
# #
# #     if not os.path.isdir(BASE_PATH):
# #         raise FileNotFoundError(f"Dataset folder not found at {BASE_PATH}")
# #
# #     for idx, folder_name in enumerate(subfolders):
# #         label_map[folder_name] = idx
# #         folder_path = os.path.join(BASE_PATH, folder_name, "images")
# #         if not os.path.isdir(folder_path):
# #             print(f"WARNING: No directory => {folder_path}")
# #             continue
# #
# #         file_list = [f for f in os.listdir(folder_path)
# #                      if f.lower().endswith(('.jpg','.jpeg','.png'))]
# #         file_list = file_list[:MAX_IMAGES_PER_CLASS]  # sample a small subset
# #         for f in file_list:
# #             all_data.append((os.path.join(folder_path, f), idx))
# #         print(f"  {folder_name} => took {len(file_list)} images (limit={MAX_IMAGES_PER_CLASS})")
# #
# #     df = pd.DataFrame(all_data, columns=['filepath','label'])
# #     print(f"\nTotal images used: {len(df)}")
# #     print(df.head())
# #
# #     # 2) Load images w/ OpenCV
# #     print("\n=== Step 2: Loading images into arrays (X, Y) ===")
# #     X_list, Y_list = [], []
# #     for i in range(len(df)):
# #         fpath, lbl = df.iloc[i]['filepath'], df.iloc[i]['label']
# #         img_bgr = cv2.imread(fpath)
# #         if img_bgr is None:
# #             print(f"Could not load: {fpath}")
# #             continue
# #         img_bgr = cv2.resize(img_bgr, TARGET_IMG_SIZE)
# #         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# #         X_list.append(img_rgb)
# #         Y_list.append(lbl)
# #     X = np.array(X_list)
# #     Y = np.array(Y_list)
# #     print("X shape:", X.shape, "| Y shape:", Y.shape)
# #
# #     # 3) Split data
# #     print("\n=== Step 3: train/val/test split ===")
# #     X_trainval, X_test, Y_trainval, Y_test = train_test_split(
# #         X, Y, test_size=TEST_RATIO, stratify=Y, random_state=42
# #     )
# #     X_train, X_val, Y_train, Y_val = train_test_split(
# #         X_trainval, Y_trainval,
# #         test_size=VAL_RATIO_OF_TRAIN,
# #         stratify=Y_trainval,
# #         random_state=42
# #     )
# #     print(f"  Train: {X_train.shape}, {Y_train.shape}")
# #     print(f"  Val:   {X_val.shape},   {Y_val.shape}")
# #     print(f"  Test:  {X_test.shape},  {Y_test.shape}")
# #
# #     # 4) Build models
# #     print("\n=== Step 4: Building CNN & VGG16 ===")
# #     cnn_model = build_functional_cnn(input_shape=(224,224,3), num_classes=NUM_CLASSES)
# #     vgg16_model = build_functional_vgg16(input_shape=(224,224,3), num_classes=NUM_CLASSES)
# #
# #     # 5) Train quickly
# #     print("\n=== Step 5: Quick training, 1 epoch each ===")
# #     from train_utils import build_dataset_via_generator
# #     train_ds_cnn_noadv = build_dataset_via_generator(X_train, Y_train, BATCH_SIZE, True, True)
# #     val_ds_cnn_noadv   = build_dataset_via_generator(X_val,   Y_val,   BATCH_SIZE, False, False)
# #
# #     start_time = time.time()
# #     cnn_model.fit(train_ds_cnn_noadv, epochs=EPOCHS, validation_data=val_ds_cnn_noadv)
# #     time_cnn_noadv = time.time() - start_time
# #
# #     train_ds_vgg_noadv = build_dataset_via_generator(X_train, Y_train, BATCH_SIZE, True, True)
# #     val_ds_vgg_noadv   = build_dataset_via_generator(X_val,   Y_val,   BATCH_SIZE, False, False)
# #
# #     start_time = time.time()
# #     vgg16_model.fit(train_ds_vgg_noadv, epochs=EPOCHS, validation_data=val_ds_vgg_noadv)
# #     time_vgg_noadv = time.time() - start_time
# #
# #     # 6) Evaluate
# #     print("\n=== Step 6: Evaluate on test set ===")
# #     from train_utils import build_test_dataset_via_generator
# #     test_ds = build_test_dataset_via_generator(X_test, Y_test, batch_size=BATCH_SIZE)
# #
# #     cnn_loss, cnn_acc = cnn_model.evaluate(test_ds, verbose=0)
# #     print(f"  CNN(no adv) => Test Acc={cnn_acc:.4f}, Loss={cnn_loss:.4f}")
# #     vgg_loss, vgg_acc = vgg16_model.evaluate(test_ds, verbose=0)
# #     print(f"  VGG(no adv) => Test Acc={vgg_acc:.4f}, Loss={vgg_loss:.4f}")
# #
# #     # 7) Summaries & Saliency
# #     print("\n=== Step 7: Summaries & Saliency ===")
# #     time_cnn_adv   = 0.0
# #     time_vgg_adv   = 0.0
# #     acc_cnn_adv    = 0.0
# #     acc_vgg_adv    = 0.0
# #     confdrop_cnn_noadv = 0.0
# #     confdrop_cnn_adv   = 0.0
# #     confdrop_vgg_noadv = 0.0
# #     confdrop_vgg_adv   = 0.0
# #     cnn_adv_model  = None
# #     vgg16_adv_model= None
# #
# #     final_results_and_saliency(
# #         time_cnn_noadv, time_cnn_adv,
# #         time_vgg_noadv, time_vgg_adv,
# #         cnn_acc, acc_cnn_adv,
# #         vgg_acc, acc_vgg_adv,
# #         confdrop_cnn_noadv, confdrop_cnn_adv,
# #         confdrop_vgg_noadv, confdrop_vgg_adv,
# #         cnn_model, cnn_adv_model,
# #         vgg16_model, vgg16_adv_model,
# #         X_test, Y_test
# #     )
# #
# #     print("\nAll done. Check for saved images: accuracy_bar.png, confdrop_bar.png, saliency_overlay.png")
# #
# # # Execute main if run as script
# # if __name__ == "__main__":
# #     main()
#
#
# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
# """
# main.py: A single-file pipeline with:
# 1) Data loading (limit images for quick test)
# 2) CNN & VGG16 (both no adv & adv)
# 3) Training, evaluating (accuracy, F1, confusion matrix)
# 4) Grad-CAM, IG, Integrated Grad-CAM
# 5) Bar plots + big saliency figure
#
# Adjust EPOCHS / MAX_PER_CLASS as needed for a faster or more thorough run.
# """
#
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
#
# import os
# import time
# import random
# import numpy as np
# import pandas as pd
# import cv2
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, f1_score
#
# import tensorflow as tf
# from tensorflow.keras import Model, Input, layers
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
#
# try:
#     from tf_explain.core.grad_cam import GradCAM
#     from tf_explain.core.integrated_gradients import IntegratedGradients
# except ImportError:
#     GradCAM = None
#     IntegratedGradients = None
#     print("Warning: tf-explain not installed => Grad-CAM & IG won't run.")
#
#
# print("All libraries imported successfully!")
#
# ###############################################################################
# # CONFIG
# ###############################################################################
# BASE_PATH = "data/COVID-19_Radiography_Dataset"
# subfolders = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
# NUM_CLASSES = 4
# TARGET_IMG_SIZE = (224, 224)
#
# TEST_RATIO = 0.15
# VAL_RATIO_OF_TRAIN = 0.1765
# BATCH_SIZE = 16
# EPOCHS = 1  # Quick test
# MAX_PER_CLASS = 50  # load only up to 50 images per class (speed)
#
# ###############################################################################
# # STEP A: Data collection & building X,Y
# ###############################################################################
# def collect_filepaths():
#     """
#     Returns a DataFrame with 'filepath','label',
#     limiting to MAX_PER_CLASS per subfolder for a quick run.
#     """
#     if not os.path.isdir(BASE_PATH):
#         raise FileNotFoundError(f"Dataset not found => {BASE_PATH}")
#
#     all_data = []
#     label_map = {}
#
#     for idx, folder_name in enumerate(subfolders):
#         label_map[folder_name] = idx
#         folder_path = os.path.join(BASE_PATH, folder_name, "images")
#         if not os.path.isdir(folder_path):
#             print(f"WARNING: skipping => {folder_path}")
#             continue
#
#         file_list = sorted(os.listdir(folder_path))
#         # filter images only
#         file_list = [fname for fname in file_list if fname.lower().endswith(('.jpg','.jpeg','.png'))]
#         # limit to max
#         file_list = file_list[:MAX_PER_CLASS]
#
#         for fname in file_list:
#             fullp = os.path.join(folder_path, fname)
#             all_data.append((fullp, idx))
#         print(f"{folder_name} => took {len(file_list)} images (limit={MAX_PER_CLASS})")
#
#     df = pd.DataFrame(all_data, columns=['filepath','label'])
#     print(f"\nTotal images used: {len(df)}")
#     return df
#
# def build_xy_arrays(df):
#     """
#     Load images from df['filepath'] into X, Y.
#     """
#     X_list, Y_list = [], []
#     for i in range(len(df)):
#         fpath, lbl = df.iloc[i]['filepath'], df.iloc[i]['label']
#         img_bgr = cv2.imread(fpath)
#         if img_bgr is None:
#             print("Could not load:", fpath)
#             continue
#         img_bgr = cv2.resize(img_bgr, TARGET_IMG_SIZE)
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         X_list.append(img_rgb)
#         Y_list.append(lbl)
#     X = np.array(X_list)
#     Y = np.array(Y_list)
#     return X, Y
#
# # Albumentations
# import albumentations as A
# augment = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.5),
#     A.Rotate(limit=15, p=0.5)
# ])
#
# def augment_image(img_255):
#     augmented = augment(image=img_255)
#     aug_img = augmented['image']
#     aug_img = aug_img.astype('float32') / 255.0
#     return aug_img
#
# def preprocess_image(img_255):
#     return img_255.astype('float32')/255.0
#
# def python_generator(X_data, Y_data, augment_flag=True, shuffle=True):
#     idxs = np.arange(len(X_data))
#     if shuffle:
#         np.random.shuffle(idxs)
#     for i in idxs:
#         img_255 = X_data[i]
#         lbl = Y_data[i]
#         if augment_flag:
#             yield augment_image(img_255), lbl
#         else:
#             yield preprocess_image(img_255), lbl
#
# def build_dataset(X_data, Y_data, batch_size=16, augment_flag=True, shuffle=True):
#     def gen():
#         yield from python_generator(X_data, Y_data, augment_flag=augment_flag, shuffle=shuffle)
#     ds = tf.data.Dataset.from_generator(
#         gen,
#         output_types=(tf.float32, tf.int32),
#         output_shapes=((224,224,3),())
#     )
#     ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return ds
#
# ###############################################################################
# # STEP B: Model Definitions
# ###############################################################################
# def build_cnn(input_shape=(224,224,3), num_classes=4):
#     inp = Input(shape=input_shape)
#     x = layers.Conv2D(64, (3,3), activation='relu')(inp)
#     x = layers.MaxPooling2D((2,2))(x)
#     x = layers.Dropout(0.3)(x)
#
#     x = layers.Conv2D(64, (3,3), activation='relu')(x)
#     x = layers.MaxPooling2D((2,2))(x)
#     x = layers.Dropout(0.3)(x)
#
#     x = layers.Conv2D(128, (3,3), activation='relu', name='cnn_last_conv')(x)
#     x = layers.MaxPooling2D((2,2))(x)
#     x = layers.Dropout(0.3)(x)
#
#     x = layers.Flatten()(x)
#     x = layers.Dense(128, activation='relu')(x)
#     x = layers.Dropout(0.3)(x)
#     out = layers.Dense(num_classes)(x)  # logits
#
#     model = Model(inp, out, name="custom_cnn")
#     model.compile(optimizer=Adam(),
#                   loss=SparseCategoricalCrossentropy(from_logits=True),
#                   metrics=['accuracy'])
#     return model
#
# def build_vgg16(input_shape=(224,224,3), num_classes=4):
#     base = VGG16(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max')
#     x = base.output
#     out = layers.Dense(num_classes)(x)
#     model = Model(base.input, out, name="vgg16_custom")
#     model.compile(optimizer=Adam(),
#                   loss=SparseCategoricalCrossentropy(from_logits=True),
#                   metrics=['accuracy'])
#     return model
#
# ###############################################################################
# # STEP C: Adversarial
# ###############################################################################
# def create_adversarial_pattern(model, x_in, y_in):
#     with tf.GradientTape() as tape:
#         tape.watch(x_in)
#         preds = model(x_in, training=False)
#         loss = SparseCategoricalCrossentropy(from_logits=True)(y_in, preds)
#     grads = tape.gradient(loss, x_in)
#     return tf.sign(grads)
#
# def adv_generator(X_data, Y_data, model, epsilon=0.1, shuffle=True):
#     idxs = np.arange(len(X_data))
#     if shuffle:
#         np.random.shuffle(idxs)
#     for i in idxs:
#         img_255 = X_data[i]
#         lbl = Y_data[i]
#         # Albumentations
#         c_img = augment_image(img_255)
#         c_tf  = tf.expand_dims(c_img, axis=0)
#         l_tf  = tf.constant([lbl], dtype=tf.int32)
#
#         sign_grad = create_adversarial_pattern(model, c_tf, l_tf)
#         adv_img = tf.clip_by_value(c_tf + epsilon*sign_grad, 0.0, 1.0)
#         yield tf.concat([c_tf, adv_img], axis=0), tf.concat([l_tf,l_tf], axis=0)
#
# def build_adv_dataset(X_data, Y_data, model, epsilon=0.1, shuffle=True, batch_size=1):
#     def gen():
#         yield from adv_generator(X_data, Y_data, model, epsilon, shuffle)
#     ds = tf.data.Dataset.from_generator(
#         gen,
#         output_types=(tf.float32, tf.int32),
#         output_shapes=((2,224,224,3),(2,))
#     )
#     ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return ds
#
# def adv_train_loop(model, X_data, Y_data, epochs=1, epsilon=0.1):
#     optimizer = model.optimizer
#     for e in range(epochs):
#         print(f"\nAdversarial Epoch {e+1}/{epochs}")
#         adv_ds = build_adv_dataset(X_data, Y_data, model, epsilon, True, 1)
#         step_idx = 0
#         for combined_imgs, combined_lbls in adv_ds:
#             combined_imgs = tf.reshape(combined_imgs, [-1,224,224,3])
#             combined_lbls = tf.reshape(combined_lbls, [-1])
#             with tf.GradientTape() as tape:
#                 logits = model(combined_imgs, training=True)
#                 loss = SparseCategoricalCrossentropy(from_logits=True)(combined_lbls, logits)
#             grads = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))
#             step_idx+=1
#             if step_idx%100==0:
#                 print(f"  step={step_idx}, loss={loss.numpy():.4f}")
#
# ###############################################################################
# # STEP D: Integrated Grad-CAM approx
# ###############################################################################
# def integrated_gradcam(model, img_rgb, class_idx, steps=20, layer_name='cnn_last_conv'):
#     """
#     Approx approach: we do Grad-CAM on alpha-interpolated images from baseline to input,
#     then average.
#     """
#     if GradCAM is None:
#         return None
#     baseline = np.zeros_like(img_rgb)
#     gradcam_ = GradCAM()
#     h, w = img_rgb.shape[:2]
#     combined_map = np.zeros((h,w), dtype=np.float32)
#
#     for s in range(steps):
#         alpha = s/float(steps)
#         interp = baseline + alpha*(img_rgb - baseline)
#         heatmap = gradcam_.explain(
#             validation_data=(np.expand_dims(interp,0), None),
#             model=model,
#             layer_name=layer_name,
#             class_index=class_idx
#         )
#         if heatmap.ndim==3 and heatmap.shape[-1]==3:
#             heatmap = np.mean(heatmap,axis=-1)
#         combined_map += heatmap
#
#     combined_map /= steps
#     return combined_map
#
# ###############################################################################
# # STEP E: Helper for final F1 & confusion matrix
# ###############################################################################
# def measure_f1_and_confusion(model, X_data, Y_data, batch_size=16):
#     ds = build_dataset(X_data, Y_data, batch_size, augment_flag=False, shuffle=False)
#     y_true = []
#     y_pred = []
#     for xb, yb in ds:
#         preds = model(xb, training=False)
#         preds_label = tf.argmax(preds, axis=1).numpy()
#         y_pred.extend(preds_label)
#         y_true.extend(yb.numpy())
#
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     conf_mat = confusion_matrix(y_true,y_pred)
#     f1_ = f1_score(y_true, y_pred, average='macro')
#     return f1_, conf_mat
#
# ###############################################################################
# # MAIN
# ###############################################################################
# def main():
#     print("\n=== Step 1: Collect filepaths (limit to MAX_PER_CLASS) ===")
#     df = collect_filepaths()
#     print(df.head())
#
#     print("\n=== Step 2: Load images => X, Y ===")
#     X, Y = build_xy_arrays(df)
#     print("X shape=", X.shape, ", Y shape=", Y.shape)
#
#     print("\n=== Step 3: train/val/test split ===")
#     X_trainval, X_test, Y_trainval, Y_test = train_test_split(X,Y,test_size=TEST_RATIO,stratify=Y,random_state=42)
#     X_train, X_val, Y_train, Y_val = train_test_split(X_trainval,Y_trainval,
#         test_size=VAL_RATIO_OF_TRAIN,stratify=Y_trainval,random_state=42)
#     print(f" Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
#
#     # Build 4 variants:
#     print("\n=== Building CNN(no adv), CNN(adv), VGG(no adv), VGG(adv) ===")
#     cnn_noadv = build_cnn()
#     vgg_noadv = build_vgg16()
#     cnn_adv   = build_cnn()
#     vgg_adv   = build_vgg16()
#
#     # Train CNN(no adv)
#     print("\n=== Training CNN(no adv) ===")
#     ds_train_cnn_noadv = build_dataset(X_train, Y_train, BATCH_SIZE, True, True)
#     ds_val_cnn_noadv   = build_dataset(X_val,   Y_val,   BATCH_SIZE, False, False)
#     t0 = time.time()
#     cnn_noadv.fit(ds_train_cnn_noadv, epochs=EPOCHS, validation_data=ds_val_cnn_noadv)
#     time_cnn_noadv = time.time() - t0
#
#     # Clone its weights to cnn_adv
#     cnn_adv.set_weights(cnn_noadv.get_weights())
#
#     # Train VGG(no adv)
#     print("\n=== Training VGG(no adv) ===")
#     ds_train_vgg_noadv = build_dataset(X_train, Y_train, BATCH_SIZE, True, True)
#     ds_val_vgg_noadv   = build_dataset(X_val,   Y_val,   BATCH_SIZE, False, False)
#     t0 = time.time()
#     vgg_noadv.fit(ds_train_vgg_noadv, epochs=EPOCHS, validation_data=ds_val_vgg_noadv)
#     time_vgg_noadv = time.time() - t0
#
#     # Clone to vgg_adv
#     vgg_adv.set_weights(vgg_noadv.get_weights())
#
#     # Adversarial training => 1 epoch each for demonstration
#     print("\n=== Adversarial training CNN(adv) for 1 epoch ===")
#     t0 = time.time()
#     adv_train_loop(cnn_adv, X_train, Y_train, epochs=1, epsilon=0.1)
#     time_cnn_adv = time.time() - t0
#
#     print("\n=== Adversarial training VGG(adv) for 1 epoch ===")
#     t0 = time.time()
#     adv_train_loop(vgg_adv, X_train, Y_train, epochs=1, epsilon=0.1)
#     time_vgg_adv = time.time() - t0
#
#     # Evaluate all 4
#     print("\n=== Evaluate all 4 on test set: measure F1, confusion, compute accuracy from conf mat ===")
#     # 1) CNN(no adv)
#     f1_cnn_noadv, cm_cnn_noadv = measure_f1_and_confusion(cnn_noadv, X_test, Y_test)
#     acc_cnn_noadv = (cm_cnn_noadv.diagonal().sum()/cm_cnn_noadv.sum())
#     print(f" CNN(no adv): ACC={acc_cnn_noadv:.3f}, F1={f1_cnn_noadv:.3f}\nConf:\n{cm_cnn_noadv}")
#
#     # 2) CNN(adv)
#     f1_cnn_adv, cm_cnn_adv = measure_f1_and_confusion(cnn_adv, X_test, Y_test)
#     acc_cnn_adv = (cm_cnn_adv.diagonal().sum()/cm_cnn_adv.sum())
#     print(f" CNN(adv): ACC={acc_cnn_adv:.3f}, F1={f1_cnn_adv:.3f}\nConf:\n{cm_cnn_adv}")
#
#     # 3) VGG(no adv)
#     f1_vgg_noadv, cm_vgg_noadv = measure_f1_and_confusion(vgg_noadv, X_test, Y_test)
#     acc_vgg_noadv = (cm_vgg_noadv.diagonal().sum()/cm_vgg_noadv.sum())
#     print(f" VGG(no adv): ACC={acc_vgg_noadv:.3f}, F1={f1_vgg_noadv:.3f}\nConf:\n{cm_vgg_noadv}")
#
#     # 4) VGG(adv)
#     f1_vgg_adv, cm_vgg_adv = measure_f1_and_confusion(vgg_adv, X_test, Y_test)
#     acc_vgg_adv = (cm_vgg_adv.diagonal().sum()/cm_vgg_adv.sum())
#     print(f" VGG(adv): ACC={acc_vgg_adv:.3f}, F1={f1_vgg_adv:.3f}\nConf:\n{cm_vgg_adv}")
#
#     # Summaries
#     df_results = pd.DataFrame([
#         ["CNN(no adv)", time_cnn_noadv, acc_cnn_noadv, f1_cnn_noadv],
#         ["CNN(adv)",    time_cnn_adv,   acc_cnn_adv,   f1_cnn_adv],
#         ["VGG(no adv)", time_vgg_noadv, acc_vgg_noadv, f1_vgg_noadv],
#         ["VGG(adv)",    time_vgg_adv,   acc_vgg_adv,   f1_vgg_adv]
#     ], columns=["Model","TrainTime(s)","Acc","F1"])
#     print("\n=== Final Table ===")
#     print(df_results)
#
#     # bar plots
#     plt.figure(figsize=(6,4))
#     plt.bar(df_results["Model"], df_results["Acc"], color=["blue","blue","green","green"])
#     plt.ylim([0,1])
#     plt.title("Accuracy of 4 Variants")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("acc_bar.png")
#     plt.show()
#
#     plt.figure(figsize=(6,4))
#     plt.bar(df_results["Model"], df_results["F1"], color=["red","red","cyan","cyan"])
#     plt.ylim([0,1])
#     plt.title("F1 of 4 Variants")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("f1_bar.png")
#     plt.show()
#
#     # Saliency: We'll pick 1 random test image, do (Grad-CAM, IG, Integrated Grad-CAM) for each model
#     if GradCAM is None or IntegratedGradients is None:
#         print("\n** Some XAI methods not installed => skipping saliency. **")
#         return
#
#     idx_ = random.randint(0,len(X_test)-1)
#     sample_img_rgb = X_test[idx_].astype(np.float32)/255.0
#     sample_label   = Y_test[idx_]
#     print(f"\n=== SALIENCY on test idx={idx_}, label={sample_label} ===")
#
#     # We'll do a 4 x 3 grid => row per model, col => GradCAM, IG, IG-CAM
#     fig, axs = plt.subplots(4,4, figsize=(18,20))
#     fig.suptitle("Saliency: Grad-CAM / IG / Integrated Grad-CAM", fontsize=16)
#
#     # We'll define a small helper:
#     def get_pred_class(model, img_rgb):
#         preds = model(np.expand_dims(img_rgb,axis=0))
#         return int(tf.argmax(preds[0]).numpy())
#
#     variants = [
#         ("CNN(no adv)", cnn_noadv, "cnn_last_conv"),
#         ("CNN(adv)",    cnn_adv,   "cnn_last_conv"),
#         ("VGG(no adv)", vgg_noadv, "block5_conv3"),
#         ("VGG(adv)",    vgg_adv,   "block5_conv3")
#     ]
#
#     for row_idx, (mname, mobj, layer_name) in enumerate(variants):
#         # col0 => original
#         axs[row_idx,0].imshow((sample_img_rgb*255).astype(np.uint8))
#         axs[row_idx,0].set_title(f"{mname}\nOriginal")
#         axs[row_idx,0].axis('off')
#
#         if mobj is None:
#             # If adv is None => skip
#             for c_ in [1,2,3]:
#                 axs[row_idx,c_].set_title(f"{mname}\nNo model??")
#                 axs[row_idx,c_].axis('off')
#             continue
#
#         pred_class = get_pred_class(mobj, sample_img_rgb)
#
#         # Grad-CAM => col1
#         gradcam_ = GradCAM()
#         hmap_gc = gradcam_.explain(
#             validation_data=(np.expand_dims(sample_img_rgb,0), None),
#             model=mobj,
#             layer_name=layer_name,
#             class_index=pred_class
#         )
#         if hmap_gc.ndim==3 and hmap_gc.shape[-1]==3:
#             hmap_gc = np.mean(hmap_gc, axis=-1)
#         axs[row_idx,1].imshow((sample_img_rgb*255).astype(np.uint8), alpha=0.6)
#         axs[row_idx,1].imshow(cv2.resize(hmap_gc, (224,224)), cmap='jet', alpha=0.4)
#         axs[row_idx,1].set_title(f"{mname}\nGrad-CAM\nclass={pred_class}")
#         axs[row_idx,1].axis('off')
#
#         # IG => col2
#         ig_ = IntegratedGradients()
#         ig_map = ig_.explain(
#             (np.expand_dims(sample_img_rgb,0), None),
#             model=mobj,
#             class_index=pred_class
#         )
#         axs[row_idx,2].imshow((sample_img_rgb*255).astype(np.uint8), alpha=0.6)
#         axs[row_idx,2].imshow(cv2.resize(ig_map, (224,224)), cmap='cool', alpha=0.4)
#         axs[row_idx,2].set_title(f"{mname}\nIG\nclass={pred_class}")
#         axs[row_idx,2].axis('off')
#
#         # Integrated Grad-CAM => col3
#         igcam_map = integrated_gradcam(mobj, sample_img_rgb, pred_class, 20, layer_name)
#         axs[row_idx,3].imshow((sample_img_rgb*255).astype(np.uint8), alpha=0.6)
#         if igcam_map is not None:
#             axs[row_idx,3].imshow(cv2.resize(igcam_map,(224,224)), cmap='plasma', alpha=0.4)
#         axs[row_idx,3].set_title(f"{mname}\nIG-CAM\nclass={pred_class}")
#         axs[row_idx,3].axis('off')
#
#     plt.tight_layout()
#     plt.savefig("all_saliency_methods.png")
#     plt.show()
#
#     print("\nDONE. Check images: acc_bar.png, f1_bar.png, all_saliency_methods.png, etc.")
#
# if __name__=="__main__":
#     main()

# main.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Local modules
from config import (
    BASE_PATH, SUBFOLDERS, MAX_IMAGES_PER_CLASS, TARGET_IMG_SIZE,
    NUM_CLASSES, TEST_RATIO, VAL_RATIO_OF_TRAIN,
    BATCH_SIZE, EPOCHS, ACC_TARGETS, RESULTS_CSV, SAL_FOLDER
)
from data_utils import (
    collect_filepaths, load_images_to_arrays
)
from train_utils import (
    build_dataset, TimeToAccCallback
)
from adv_utils import adv_train_loop
from model_defs import build_cnn, build_vgg16
from metrics_utils import measure_f1_and_confusion, integrated_gradcam
from xai_utils import do_saliency_methods

from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.integrated_gradients import IntegratedGradients

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

print("All libraries & local modules imported successfully!")

def main():
    # Step 1: Collect limited filepaths
    df = collect_filepaths()  # up to MAX_IMAGES_PER_CLASS from each folder
    print(df.head())

    # Step 2: Build X, Y
    X, Y = load_images_to_arrays(df)
    print("X shape:", X.shape, " | Y shape:", Y.shape)

    # Step 3: train/val/test split
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y, test_size=TEST_RATIO, stratify=Y, random_state=42
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval,
        test_size=VAL_RATIO_OF_TRAIN, stratify=Y_trainval, random_state=42
    )
    print(f"Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # Build 4 model variants
    cnn_noadv = build_cnn()
    cnn_adv   = build_cnn()  # we’ll copy weights from cnn_noadv later
    vgg_noadv = build_vgg16()
    vgg_adv   = build_vgg16()

    # Prepare callbacks for each model => measure time to 85% & 90% accuracy
    cb_cnn_noadv = TimeToAccCallback(acc_targets=ACC_TARGETS)
    cb_vgg_noadv = TimeToAccCallback(acc_targets=ACC_TARGETS)
    cb_cnn_adv   = TimeToAccCallback(acc_targets=ACC_TARGETS)
    cb_vgg_adv   = TimeToAccCallback(acc_targets=ACC_TARGETS)

    # ============ Steps-per-epoch for repeating DS ============
    train_size = len(X_train)
    val_size   = len(X_val)
    steps_per_epoch      = max(1, train_size // BATCH_SIZE)
    validation_steps     = max(1, val_size // BATCH_SIZE)

    # Step 4: Train CNN(no adv)
    print("\n=== Training CNN(no adv) ===")
    ds_train_cnn_noadv = build_dataset(X_train, Y_train, batch_size=BATCH_SIZE, augment_flag=True, shuffle=True, repeat=True)
    ds_val_cnn_noadv   = build_dataset(X_val,   Y_val,   batch_size=BATCH_SIZE, augment_flag=False,shuffle=False,repeat=True)
    t0 = time.time()
    cnn_noadv.fit(
        ds_train_cnn_noadv,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_val_cnn_noadv,
        validation_steps=validation_steps,
        callbacks=[cb_cnn_noadv, EarlyStopping(patience=2,monitor='val_loss')]
    )
    time_cnn_noadv = time.time() - t0

    # Step 5: Clone weights to CNN(adv), do adv train
    cnn_adv.set_weights(cnn_noadv.get_weights())
    print("\n=== Adversarial training CNN(adv) for 1 epoch demonstration ===")
    t0 = time.time()
    adv_train_loop(cnn_adv, X_train, Y_train, epochs=1, epsilon=0.1)
    time_cnn_adv = time.time() - t0

    # Step 6: Train VGG(no adv)
    print("\n=== Training VGG(no adv) ===")
    ds_train_vgg_noadv = build_dataset(X_train, Y_train, BATCH_SIZE, True, True, repeat=True)
    ds_val_vgg_noadv   = build_dataset(X_val,   Y_val,   BATCH_SIZE, False,False, repeat=True)
    t0 = time.time()
    vgg_noadv.fit(
        ds_train_vgg_noadv,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_val_vgg_noadv,
        validation_steps=validation_steps,
        callbacks=[cb_vgg_noadv, EarlyStopping(patience=2,monitor='val_loss')]
    )
    time_vgg_noadv = time.time() - t0

    # Step 7: Clone weights => VGG(adv)
    vgg_adv.set_weights(vgg_noadv.get_weights())
    print("\n=== Adversarial training VGG(adv) for 1 epoch demonstration ===")
    t0 = time.time()
    adv_train_loop(vgg_adv, X_train, Y_train, epochs=1, epsilon=0.1)
    time_vgg_adv = time.time() - t0

    # Step 8: Evaluate all 4 => measure f1, confusion => final Acc
    print("\n=== Evaluate all 4 variants on test set ===")
    # measure_f1_and_confusion => doesn't repeat ds, but internally it might, so let's just pass repeat=False or define a custom measure
    f1_cnn_noadv, cm_cnn_noadv = measure_f1_and_confusion(cnn_noadv, X_test, Y_test, build_dataset, batch_size=BATCH_SIZE)
    acc_cnn_noadv = cm_cnn_noadv.diagonal().sum()/cm_cnn_noadv.sum()
    f1_cnn_adv, cm_cnn_adv = measure_f1_and_confusion(cnn_adv, X_test, Y_test, build_dataset, batch_size=BATCH_SIZE)
    acc_cnn_adv = cm_cnn_adv.diagonal().sum()/cm_cnn_adv.sum()

    f1_vgg_noadv, cm_vgg_noadv = measure_f1_and_confusion(vgg_noadv, X_test, Y_test, build_dataset, batch_size=BATCH_SIZE)
    acc_vgg_noadv = cm_vgg_noadv.diagonal().sum()/cm_vgg_noadv.sum()
    f1_vgg_adv, cm_vgg_adv = measure_f1_and_confusion(vgg_adv, X_test, Y_test, build_dataset, batch_size=BATCH_SIZE)
    acc_vgg_adv = cm_vgg_adv.diagonal().sum()/cm_vgg_adv.sum()

    # Extract times to 85/90% from each callback
    def safe_time(d, thr):
        return d[thr] if d[thr] is not None else None

    # Build final results as a list of dicts
    results_list = [
        {
            "Model":"CNN(no adv)",
            "TrainTime": time_cnn_noadv,
            "AccFinal":  acc_cnn_noadv,
            "F1Final":   f1_cnn_noadv,
            "Time_Acc85": safe_time(cb_cnn_noadv.times_reached, 0.85),
            "Time_Acc90": safe_time(cb_cnn_noadv.times_reached, 0.90)
        },
        {
            "Model":"CNN(adv)",
            "TrainTime": time_cnn_adv,
            "AccFinal":  acc_cnn_adv,
            "F1Final":   f1_cnn_adv,
            "Time_Acc85": safe_time(cb_cnn_adv.times_reached, 0.85),
            "Time_Acc90": safe_time(cb_cnn_adv.times_reached, 0.90)
        },
        {
            "Model":"VGG(no adv)",
            "TrainTime": time_vgg_noadv,
            "AccFinal":  acc_vgg_noadv,
            "F1Final":   f1_vgg_noadv,
            "Time_Acc85": safe_time(cb_vgg_noadv.times_reached, 0.85),
            "Time_Acc90": safe_time(cb_vgg_noadv.times_reached, 0.90)
        },
        {
            "Model":"VGG(adv)",
            "TrainTime": time_vgg_adv,
            "AccFinal":  acc_vgg_adv,
            "F1Final":   f1_vgg_adv,
            "Time_Acc85": safe_time(cb_vgg_adv.times_reached, 0.85),
            "Time_Acc90": safe_time(cb_vgg_adv.times_reached, 0.90)
        },
    ]

    df_results = pd.DataFrame(results_list)
    print("\n=== Final Results Table ===")
    print(df_results)

    # Save to CSV
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"Saved results => {RESULTS_CSV}")

    # Quick bar charts for final Acc & F1
    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(6,4))
    plt.bar(df_results["Model"], df_results["AccFinal"], color=["blue","blue","green","green"])
    plt.ylim([0,1])
    plt.title("Final Accuracy for 4 Variants")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/final_acc_bar.png")
    plt.show()

    plt.figure(figsize=(6,4))
    plt.bar(df_results["Model"], df_results["F1Final"], color=["red","red","cyan","cyan"])
    plt.ylim([0,1])
    plt.title("Final F1 for 4 Variants")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/final_f1_bar.png")
    plt.show()

    # Step 9: Saliency with Grad-CAM, IG, IG-CAM (one random test image)
    if (GradCAM is None) or (IntegratedGradients is None):
        print("\n** tf-explain not installed => skipping saliency. **")
        return

    idx_ = random.randint(0, len(X_test)-1)
    sample_img = X_test[idx_].astype(np.float32)/255.0
    sample_lbl = Y_test[idx_]
    print(f"\n=== Saliency for random test idx={idx_}, label={sample_lbl} ===")

    # We'll do a 4x4 figure => row: CNN(no adv), CNN(adv), VGG(no adv), VGG(adv)
    # columns => Original, Grad-CAM, IG, IG-CAM
    fig, axs = plt.subplots(4,4, figsize=(18,20))
    fig.suptitle("Saliency: Grad-CAM, IG, IG-CAM", fontsize=16)

    def get_pred_class(model, img_rgb):
        preds = model(np.expand_dims(img_rgb,axis=0), training=False)
        return int(tf.argmax(preds[0]).numpy())

    variants = [
        ("CNN(no adv)", cnn_noadv,  "cnn_last_conv"),
        ("CNN(adv)",    cnn_adv,    "cnn_last_conv"),
        ("VGG(no adv)", vgg_noadv,  "block5_conv3"),
        ("VGG(adv)",    vgg_adv,    "block5_conv3")
    ]

    import cv2
    from metrics_utils import integrated_gradcam  # for IG-CAM
    for r, (mname, mobj, layer_name) in enumerate(variants):
        # col0 => original
        axs[r,0].imshow((sample_img*255).astype(np.uint8))
        axs[r,0].set_title(f"{mname}\nOriginal")
        axs[r,0].axis('off')

        if mobj is None:
            for c_ in [1,2,3]:
                axs[r,c_].axis('off')
                axs[r,c_].set_title(f"{mname} => None??")
            continue

        class_idx = get_pred_class(mobj, sample_img)

        # Grad-CAM => col1
        gradcam_ = GradCAM()
        gmap = gradcam_.explain(
            (np.expand_dims(sample_img,0), None),
            model=mobj,
            layer_name=layer_name,
            class_index=class_idx
        )
        if gmap.ndim==3 and gmap.shape[-1]==3:
            gmap = np.mean(gmap, axis=-1)
        axs[r,1].imshow((sample_img*255).astype(np.uint8), alpha=0.6)
        axs[r,1].imshow(cv2.resize(gmap,(224,224)), cmap='jet', alpha=0.4)
        axs[r,1].set_title(f"{mname}\nGrad-CAM c={class_idx}")
        axs[r,1].axis('off')

        # IG => col2
        ig_ = IntegratedGradients()
        igmap_ = ig_.explain(
            (np.expand_dims(sample_img,0),None),
            model=mobj,
            class_index=class_idx
        )
        axs[r,2].imshow((sample_img*255).astype(np.uint8), alpha=0.6)
        axs[r,2].imshow(cv2.resize(igmap_,(224,224)), cmap='cool', alpha=0.4)
        axs[r,2].set_title(f"{mname}\nIG c={class_idx}")
        axs[r,2].axis('off')

        # IG-CAM => col3
        igcam_map = integrated_gradcam(mobj, sample_img, class_idx, steps=20, layer_name=layer_name)
        axs[r,3].imshow((sample_img*255).astype(np.uint8), alpha=0.6)
        if igcam_map is not None:
            axs[r,3].imshow(cv2.resize(igcam_map,(224,224)), cmap='plasma', alpha=0.4)
        axs[r,3].set_title(f"{mname}\nIG-CAM c={class_idx}")
        axs[r,3].axis('off')

    plt.tight_layout()
    os.makedirs(SAL_FOLDER, exist_ok=True)
    sal_path = os.path.join(SAL_FOLDER, "all_saliency_methods.png")
    plt.savefig(sal_path)
    plt.show()

    print(f"\nAll done. Saliency figure => {sal_path}")
    print("Check CSV =>", RESULTS_CSV)

if __name__=="__main__":
    main()