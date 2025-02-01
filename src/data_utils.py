# data_utils.py
import os
import cv2
import numpy as np
import pandas as pd
from config import BASE_PATH, SUBFOLDERS, MAX_IMAGES_PER_CLASS, TARGET_IMG_SIZE
import albumentations as A

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5)
])

def collect_filepaths():
    """
    Returns a DataFrame with ('filepath','label'),
    limiting to MAX_IMAGES_PER_CLASS if not None.
    """
    all_data = []
    for idx, folder_name in enumerate(SUBFOLDERS):
        folder_path = os.path.join(BASE_PATH, folder_name, "images")
        if not os.path.isdir(folder_path):
            print(f"WARNING: {folder_path} not found. Skipping.")
            continue
        file_list = sorted(os.listdir(folder_path))
        # filter images
        file_list = [f for f in file_list if f.lower().endswith(('.jpg','.jpeg','.png'))]

        if MAX_IMAGES_PER_CLASS is not None:
            file_list = file_list[:MAX_IMAGES_PER_CLASS]

        for fname in file_list:
            all_data.append((os.path.join(folder_path, fname), idx))

        print(f"{folder_name} => took {len(file_list)} images. (limit={MAX_IMAGES_PER_CLASS})")

    df = pd.DataFrame(all_data, columns=['filepath','label'])
    print(f"Total images used: {len(df)}")
    return df

def load_images_to_arrays(df):
    """
    Load images from disk into X, Y (NumPy).
    """
    X_list, Y_list = [], []
    for i in range(len(df)):
        fpath, lbl = df.iloc[i]['filepath'], df.iloc[i]['label']
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            print("Could not load:", fpath)
            continue
        img_bgr = cv2.resize(img_bgr, TARGET_IMG_SIZE)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        X_list.append(img_rgb)
        Y_list.append(lbl)
    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y

def augment_image(img_255):
    # Albumentations augment
    augmented = augment(image=img_255)
    aug_img = augmented['image'].astype(np.float32)/255.0
    return aug_img

def preprocess_image(img_255):
    return img_255.astype(np.float32)/255.0