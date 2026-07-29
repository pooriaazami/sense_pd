import os
import sys
import argparse
from datetime import datetime
from functools import partial
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from utils import get_config
from models import MotionAGFormer
from data import Motion3DDataset
from train import RandomFrameMask, RandomJointMask
from utils import motion_loss_fn, joints_loss_fn

def pretext_loss(predicted_joints, ground_truth, mask, lambd):
    motion_loss = motion_loss_fn(predicted_joints, ground_truth, mask)
    joints_loss = joints_loss_fn(predicted_joints, ground_truth, mask)
    
    return lambd * motion_loss + joints_loss, motion_loss, joints_loss

def main():
    ...