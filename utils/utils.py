from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os, ssl
import urllib.request

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def download_pretrained(url, dir, filename=None):
    # Check if file already exists
    if filename is None:
        filename = os.path.basename(url)
    if os.path.exists(dir+filename):
        return
    if not os.path.exists(dir):
        os.makedirs(dir)

    try:
        # Create a URL opener that skips SSL check
        opener = urllib.request.build_opener()
        opener.add_handler(urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))

        # Download the file
        with opener.open(url) as response:
            file_contents = response.read()
    except Exception:
        raise RuntimeError(f"Failed to download pretrained weights, please manually download it from {url} and save it as {dir+filename}")

    # Save the file to the specified directory
    save_path = os.path.join(dir, filename)
    with open(save_path, 'wb') as out_file:
        out_file.write(file_contents)