# metrics_utils.py
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
from tf_explain.core.grad_cam import GradCAM

def measure_f1_and_confusion(model, X_data, Y_data, build_dataset_fn, batch_size=16):
    ds = build_dataset_fn(X_data, Y_data, batch_size, augment_flag=False, shuffle=False)
    y_true, y_pred = [], []
    for xb, yb in ds:
        preds = model(xb, training=False)
        preds_label = tf.argmax(preds,axis=1).numpy()
        y_pred.extend(preds_label)
        y_true.extend(yb.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    conf = confusion_matrix(y_true, y_pred)
    f1_  = f1_score(y_true, y_pred, average='macro')
    return f1_, conf

def integrated_gradcam(model, img_rgb, class_idx, steps=20, layer_name='cnn_last_conv'):
    """
    Simple approximate approach: linearly interpolate from baseline (zero) to input,
    do Grad-CAM each step, average the heatmaps.
    """
    gradcam_ = GradCAM()
    h, w = img_rgb.shape[:2]
    baseline = np.zeros_like(img_rgb)
    combined_map = np.zeros((h,w),dtype=np.float32)
    for s in range(steps):
        alpha = s/float(steps)
        interp_ = baseline + alpha*(img_rgb - baseline)
        heatmap = gradcam_.explain(
            validation_data=(np.expand_dims(interp_,axis=0),None),
            model=model,
            layer_name=layer_name,
            class_index=class_idx
        )
        # if shape is (H,W,3), convert to grayscale
        if heatmap.ndim==3 and heatmap.shape[-1]==3:
            heatmap = np.mean(heatmap, axis=-1)
        combined_map += heatmap
    combined_map /= steps
    return combined_map