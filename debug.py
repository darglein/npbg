import argparse
import threading
import yaml
import re
import lpips
from glumpy import app, gloo, glm, gl, transforms
from glumpy.ext import glfw
from torchvision import transforms
from npbg.datasets.common import load_image, ToTensor
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
from npbg.utils.train import load_model_checkpoint


class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])
    def forward(self, x):
        return x

def test_unet():
    net = UNet(
        num_input_channels=[8, 8, 8, 8, 8],
        # num_input_channels=[8],
        num_output_channels=3,
        feature_scale=4,
        more_layers=0,
        upsample_mode='bilinear',
        norm_layer='bn',
        last_act='',
        conv_block="gated"
    )

    load_model_checkpoint(
        "/home/dari/Projects/pointrendering2/Code/External/npbg/data/logs/04-06_10-03-11/checkpoints/UNet_stage_0_epoch_4_net.pth",
        net)

    inpu = (torch.ones((1, 8, 512, 512)), torch.ones((1, 8, 256, 256)), torch.ones((1, 8, 128, 128)),
            torch.ones((1, 8, 64, 64)), torch.ones((1, 8, 32, 32)))


    net.eval()
    result = net.forward(inpu)
    PrintTensorInfo(result)


    container = torch.jit.script(Container(net.state_dict()))
    container.save("pytorch_net.pt")

    # texture
    ckpt = torch.load("/home/dari/Projects/pointrendering2/Code/External/npbg/data/logs/04-06_10-03-11/checkpoints/PointTexture_stage_0_epoch_4_church_anime.pth", map_location='cpu')['state_dict']
    container = torch.jit.script(Container(ckpt))
    container.save("pytorch_texture.pt")


def test_vgg():
    loss = VGGLoss('caffe')
    loss.eval()

    input = torch.ones((1, 3, 128, 128)) * 0.3
    target = torch.ones_like(input) * 0.6
    # print(input.shape)

    result = loss.forward(input, target)
    PrintTensorInfo(result)
    # print(result)

    # exit(

    # model = torchvision.models.vgg19(pretrained=True).features
    model = loss.vgg19
    torch.jit.save(torch.jit.script(model), 'vgg_script_caffe.pth')

def test_lpips():
    print("test_lpips")

    loss_fn_alex = lpips.LPIPS(net='alex')
    example = torch.rand(1, 3, 224, 224)
    print("Trace")
    traced_script_module = torch.jit.trace(loss_fn_alex, (example, example))
    traced_script_module.save("traced_lpips.pt")



    default_target_transform = torchvision.transforms.Compose([
        ToTensor(),
    ])


    image0 = load_image("/home/dari/Projects/pointrendering2/Code/data/test/out.jpg")
    image1 = load_image("/home/dari/Projects/pointrendering2/Code/data/test/tar.jpg")

    img0 = default_target_transform(image0).unsqueeze(0)
    img1 = default_target_transform(image1).unsqueeze(0)



    # print("Input")
    # PrintTensorInfo(img0)
    # PrintTensorInfo(img1)

    # print(img0[0, 0, 50, 50].item())
    # print(img0[0, 1, 50, 50].item())
    # print(img0[0,2,50,50].item())
    # exit(0)
    #img0 = torch.zeros(1, 3, 64, 64)  # image should be RGB, IMPORTANT: normalized to [-1,1]
    #img1 = torch.zeros(1, 3, 64, 64)

    print("MSE", torch.nn.functional.mse_loss(img0,img1).item())

    img0 = img0 * 2 - 1
    img1 = img1 * 2 - 1

    print("LPIPS", loss_fn_alex(img0, img1).item())
    print("LPIPS (traced)", traced_script_module(img0, img1).item())




    pass

if __name__ == '__main__':
    print("debug")
    test_lpips()
    # test_partial_conv()
    #test_unet()
    # test_vgg()


