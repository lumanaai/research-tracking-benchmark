import torch
import torch.nn as nn
from argparse import Namespace
import os
from multiprocessing.dummy.connection import families
from numpy import True_

from lightweight_human_pose_estimation.pose_model import pose_model
from scripts.extract_poses_colors import extract_poses_colors
import copy

WORKDIR = os.getcwd()
RESOURCE_DIR = WORKDIR + '/../resources/weights/lightweight_human_pose_estimation/'

default_pose_args = Namespace(
    checkpoint_path = RESOURCE_DIR + 'checkpoint_iter_370000.pth',
    cpu = False,
    device = torch.device('cuda'),
    half = False
)

def parsePose(msg):
    pose_args = copy.deepcopy(default_pose_args)   #   Should not change
    return pose_args

def get_pose_args(msg):
    pose_args = parsePose(msg)   #   Should not change
    return pose_args


class cloths_model(nn.Module):
    def __init__(self, msg={}):
        super().__init__()
        self.pose_args = get_pose_args(msg)

        self.pose_model = pose_model(self.pose_args)
        self.extract_poses_colors = extract_poses_colors(self.pose_args.device)

    def forward(self, peoples):
        with torch.no_grad():
            pose_results = self.pose_model(peoples)
            pose_results = self.extract_poses_colors(pose_results)

        return pose_results