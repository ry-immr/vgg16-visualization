import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from torchvision import models
import skimage


class VGG16Features(object):
    def __init__(self, device, target_layer=16):
        self.device = device
        self.vgg16features = nn.ModuleList(list(models.vgg16(pretrained=True).features)[:target_layer]).eval()

    def visualize(self, img, save_path):
        with torch.no_grad():
            x = img.to(self.device)
            for f in self.vgg16features:
                x = f(x)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        output_features = x[0].to('cpu').detach().numpy()
            
        for idx, f in enumerate(output_features):
            skimage.io.imsave(os.path.join(save_path, '{:03}.png'.format(idx)), f)
