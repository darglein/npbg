import argparse
import threading
import yaml
import re

from glumpy import app, gloo, glm, gl, transforms
from glumpy.ext import glfw

from npbg.gl.render import OffscreenRender, create_shared_texture, cpy_tensor_to_buffer, cpy_tensor_to_texture
from npbg.gl.programs import NNScene
from npbg.gl.utils import load_scene_data, get_proj_matrix, crop_intrinsic_matrix, crop_proj_matrix, \
    setup_scene, rescale_K, FastRand, nearest_train, pca_color, extrinsics_from_view_matrix, extrinsics_from_xml
from npbg.gl.nn import OGL
from npbg.gl.camera import Trackball

import os, sys
import time
import numpy as np
import torch
import cv2
import io
from npbg.models.unet import UNet
from npbg.criterions.vgg_loss import *


def test_unet():
    net = UNet(
        num_input_channels=8,
        num_output_channels=3,
        feature_scale=4,
        more_layers=0,
        upsample_mode='bilinear',
        norm_layer='bn',
        last_act='',
        conv_block="gated"
    )
    print(net)

    input = torch.ones((1, 8, 512, 512))
    result = net.forward(input)
    PrintTensorInfo(result)


def test_vgg():
    loss = VGGLoss('pytorch')
    loss.eval()

    input = torch.ones((1, 3, 128, 128)) * 0.3
    target = torch.ones_like(input) * 0.6
    # print(input.shape)

    result = loss.forward(input, target)
    PrintTensorInfo(result)
    # print(result)

    # exit(

    model = torchvision.models.vgg19(pretrained=True).features
    torch.jit.save(torch.jit.script(model), 'vgg_script.pth')

def test_partial_conv():

    pass

if __name__ == '__main__':
    print("debug")
    # test_partial_conv()
    test_unet()


