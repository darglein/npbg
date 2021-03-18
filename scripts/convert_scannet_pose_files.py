import argparse
import glob
from PIL import Image
import numpy as np
import os
import tqdm
import cv2
import json
import pprint

join = os.path.join
basename = os.path.basename

def get_poses_scannet(n_frames):
    poses=[]
    for frame_id in range(n_frames):
        poses.append(np.loadtxt(join(args.input, 'pose', f'{frame_id}.txt')))
    return poses

def convert_pose_to_npbg(pose):
    # npbg has y and z axis inverted
    pose[:, 1:3] = -pose[:, 1:3]
    return pose

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()

    print("input dir: ", args.input)
    n_frames = len(os.listdir(join(args.input, 'color')))

    print("found ", n_frames, " frames")

    poses =  get_poses_scannet(n_frames)

    stacked = np.zeros((n_frames * 4, 4))

    for i in range(n_frames):
        stacked[i*4:(i+1)*4, :] = convert_pose_to_npbg(poses[i])

    np.savetxt(join(args.input, 'npbg_poses.txt'), stacked)
