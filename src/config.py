# config.py
import os

BASE_PATH = "data/COVID-19_Radiography_Dataset"
SUBFOLDERS = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
MAX_IMAGES_PER_CLASS = 1000  # up to 1000 per class
# MAX_IMAGES_PER_CLASS = None   # means don't limit per class
# EPOCHS = 100
 # limit per class for quick tests; set None to load all
TARGET_IMG_SIZE = (224, 224)
NUM_CLASSES = 4

TEST_RATIO = 0.15
VAL_RATIO_OF_TRAIN = 0.1765
BATCH_SIZE = 16
EPOCHS = 5          # more thorough than 1 epoch

# If you want times to a certain accuracy:
ACC_TARGETS = [0.85, 0.90]   # measure time to 85% and 90% accuracy

RESULTS_CSV = "results/final_metrics.csv"  # where to save final metrics
SAL_FOLDER  = "results/saliency"           # subfolder for saliency images
os.makedirs("results", exist_ok=True)
os.makedirs(SAL_FOLDER, exist_ok=True)