# xai_utils.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.integrated_gradients import IntegratedGradients
from metrics_utils import integrated_gradcam

def do_saliency_methods(model, img_rgb, class_idx, layer_name='cnn_last_conv'):
    """
    Returns a dict: {
      'gradcam': <heatmap 2D>,
      'ig': <heatmap 2D>,
      'igcam': <heatmap 2D>
    }
    """
    out_maps = {'gradcam': None, 'ig': None, 'igcam': None}
    # Grad-CAM
    gradcam_ = GradCAM()
    gc_map = gradcam_.explain(
        validation_data=(np.expand_dims(img_rgb,axis=0), None),
        model=model,
        layer_name=layer_name,
        class_index=class_idx
    )
    if gc_map.ndim==3 and gc_map.shape[-1]==3:
        gc_map = np.mean(gc_map, axis=-1)
    out_maps['gradcam'] = gc_map

    # IG
    ig_ = IntegratedGradients()
    ig_map = ig_.explain(
        validation_data=(np.expand_dims(img_rgb,axis=0),None),
        model=model,
        class_index=class_idx
    )
    out_maps['ig'] = ig_map

    # Integrated Grad-CAM
    igcam_map = integrated_gradcam(model, img_rgb, class_idx, layer_name=layer_name)
    out_maps['igcam'] = igcam_map

    return out_maps