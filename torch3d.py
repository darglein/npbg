import configparser
import os

import numpy
import pytorch3d
import torch
import torchvision.transforms
from PIL import Image
from torchvision.utils import save_image
# Data structures and functions for rendering
from matplotlib import pyplot as plt
from pytorch3d.io import load_ply
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

from pytorch3d.utils.camera_conversions import *

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor, PerspectiveCameras
)


def cameras_from_opencv_projection2(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
) -> PerspectiveCameras:
    """
    Converts a batch of OpenCV-conventioned cameras parametrized with the
    rotation matrices `R`, translation vectors `tvec`, and the camera
    calibration matrices `camera_matrix` to `PerspectiveCameras` in PyTorch3D
    convention.
    More specifically, the conversion is carried out such that a projection
    of a 3D shape to the OpenCV-conventioned screen of size `image_size` results
    in the same image as a projection with the corresponding PyTorch3D camera
    to the NDC screen convention of PyTorch3D.
    More specifically, the OpenCV convention projects points to the OpenCV screen
    space as follows:
        ```
        x_screen_opencv = camera_matrix @ (R @ x_world + tvec)
        ```
    followed by the homogenization of `x_screen_opencv`.
    Note:
        The parameters `R, tvec, camera_matrix` correspond to the outputs of
        `cv2.decomposeProjectionMatrix`.
        The `rvec` parameter of the `cv2.projectPoints` is an axis-angle vector
        that can be converted to the rotation matrix `R` expected here by
        calling the `so3_exp_map` function.
    Args:
        R: A batch of rotation matrices of shape `(N, 3, 3)`.
        tvec: A batch of translation vectors of shape `(N, 3)`.
        camera_matrix: A batch of camera calibration matrices of shape `(N, 3, 3)`.
        image_size: A tensor of shape `(N, 2)` containing the sizes of the images
            (height, width) attached to each camera.
    Returns:
        cameras_pytorch3d: A batch of `N` cameras in the PyTorch3D convention.
    """
    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    use_ndc = False

    if use_ndc:
        # Get the PyTorch3D focal length and principal point.
        focal_pytorch3d = focal_length / (0.5 * image_size_wh)
        p0_pytorch3d = -(principal_point / (0.5 * image_size_wh) - 1)
    else:
        focal_pytorch3d = focal_length
        p0_pytorch3d = principal_point

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1

    return PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
        image_size=image_size,
        in_ndc=use_ndc
    )
from npbg.criterions.vgg_loss import PrintTensorInfo

device = torch.device("cuda:0")


class PointScene:

    def ReadConfig(self, dir):
        config = configparser.ConfigParser()

        ########################################
        config.read(dir + "dataset.ini")
        self.image_dir = config["SceneDatasetParams"]["image_dir"]

        ########################################
        config.read(dir + "camera.ini")
        self.w = int(config["SceneCameraParams"]["w"])
        self.h = int(config["SceneCameraParams"]["h"])

        intr = numpy.array(config["SceneCameraParams"]["K"].split()).astype(float)
        self.K = torch.Tensor( [(intr[0], 0, intr[2]), (0,intr[1],intr[3]), (0,0,1)]).to(device).unsqueeze(0)


        ########################################
        with open(dir + "images.txt", "r") as myfile:
            self.image_files = myfile.readlines()

        for index, item in enumerate(self.image_files ):
            self.image_files [index] = item.strip().replace('\n','')



    def __init__(self, dir):
        print("Loading scene ", dir)
        self.ReadConfig(dir)
        a = numpy.loadtxt(dir + "poses_view_matrix.txt")
        self.num_images = a.shape[0] // 4
        self.view_matrices = []
        for i in range(self.num_images):
            V = a[i * 4:(i + 1) * 4, :]
            self.view_matrices.append(V)

        # convert view matrix to R,t
        self.camera_rs = []
        self.camera_ts = []

        for V in self.view_matrices:
            R = V[0:3, 0:3]
            t = V[0:3, 3:4]
            self.camera_rs.append(R)
            self.camera_ts.append(t.transpose())


        self.R = torch.tensor(self.camera_rs).float().to(device)
        self.T = torch.tensor(self.camera_ts).squeeze(1).float().to(device)



        self.image_size = (self.h,self.w)
        self.image_size_tensor = torch.Tensor(self.image_size).unsqueeze(0).to(device).int()

        pc = load_ply("/home/dari/Projects/pointrendering2/Code/scenes/tt_train_colmap/point_cloud.ply")
        verts = torch.Tensor(pc[0]).to(device).unsqueeze(0)
        rgb = verts.clone()
        rgb = torch.ones_like(verts)
        # PrintTensorInfo(verts)
        self.point_cloud = Pointclouds(points=verts, features=rgb)

        print("Scene loaded: R, T , K , size")
        PrintTensorInfo(self.R)
        PrintTensorInfo(self.T)
        PrintTensorInfo(self.K)
        PrintTensorInfo(self.image_size_tensor)

    def GetGroundTruth(self,index):

        trans = torchvision.transforms.ToTensor()

        name = self.image_files[index]
        image = Image.open(os.path.join(self.image_dir, name))

        return trans(image)


    def GetCamera(self, index):

        R = self.R[index:index+1,:,:]
        T = self.T[index:index+1,:]

        print("Loading camera " + str(index))
        print(R)
        print(T)
        print(self.K)
        print(self.image_size_tensor)

        #cameras = cameras_from_opencv_projection(R,T , self.K, self.image_size_tensor)
        cameras = cameras_from_opencv_projection2(R, T, self.K, self.image_size_tensor)
        cameras = cameras.to(device)
        assert len(cameras) == 1

        print("test camera")

        test = opencv_from_cameras_projection(cameras, self.image_size_tensor)
        print(test)

        return cameras


if __name__ == '__main__':
    print("torch 3d")
    scene = PointScene("/home/dari/Projects/pointrendering2/Code/scenes/tt_train_colmap_undis/")


    cameras = scene.GetCamera(1)

    # cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

    # print("Num Cameras", len(cameras))
    # assert len(cameras) == 1

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
    raster_settings = PointsRasterizationSettings(
        image_size=scene.image_size,
        # radius=0.003,
        radius=0.005,
        points_per_pixel=10
    )

    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    images = renderer(scene.point_cloud)
    PrintTensorInfo(images)

    save_image(images[0].permute((2,0,1)), "debug/img0.jpg")

    gt = scene.GetGroundTruth(1)
    print("gt")
    PrintTensorInfo(gt)
    save_image(gt, "debug/img0_gt.jpg")

    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off");
    plt.waitforbuttonpress()
