"""
Lightweight InceptionV3 feature extractor (PyTorch/torchvision).

Purpose
- Produce a 2048-dimensional ImageNet feature vector for a single BGR frame.
- Matches the training stack (torchvision InceptionV3 with global average pooling).

API
- extract_iv3_features(frame_bgr, image_size=(299, 299), device=None) -> np.ndarray (2048,)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

# Initialize a single global InceptionV3 backbone (ImageNet weights).
# The final FC is replaced with Identity so forward returns (N, 2048) features.
_iv3_weights = Inception_V3_Weights.IMAGENET1K_V1
_iv3_model = inception_v3(weights=_iv3_weights)
_iv3_model.aux_logits = False
_iv3_model.fc = nn.Identity()  # return (N, 2048)
_iv3_model.eval()
for p in _iv3_model.parameters():
    p.requires_grad = False

# ImageNet normalization constants from the loaded weights' metadata.
_IMAGENET_MEAN = torch.tensor(_iv3_weights.meta["mean"]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor(_iv3_weights.meta["std"]).view(3, 1, 1)

def extract_iv3_features(frame_bgr, image_size=(299, 299), device=None):
    """Extract a 2048-D ImageNet InceptionV3 feature vector for one frame.

    Args:
        frame_bgr: OpenCV image in BGR order, shape (H, W, 3), dtype uint8.
        image_size: Target spatial size for InceptionV3 (default (299, 299)).
        device: Optional torch.device. If None, uses CPU.

    Returns:
        Numpy array of shape (2048,) containing the feature vector (float32).
    """
    if device is None:
        device = torch.device("cpu")

    # Convert BGR → RGB and resize to the expected InceptionV3 input size.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(frame_rgb, image_size)

    # Convert to torch tensor in [0, 1] and normalize using ImageNet stats.
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = tensor.unsqueeze(0).to(device)  # [1, 3, 299, 299]

    with torch.no_grad():
        # Move the global model to the requested device on demand and run forward.
        # The model outputs a single 2048-D vector per input since fc=Identity.
        feats = _iv3_model.to(device)(tensor)  # [1, 2048]
    return feats.squeeze(0).cpu().numpy()
