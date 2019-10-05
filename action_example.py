#!/usr/bin/env python3
from __future__ import print_function

import sys
import numpy as np
from numpy import sign
from math import sin, cos, pi

import os
import os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import transforms

from PIL import Image


def load_odom(dir, ang_rel):
    diff_pose = []
    R = 0.2
    angle = 0. #for robot, use sensor reading
    max_angle = 30.0 * (pi/180)
    c = 2*(max_angle/ang_rel)   
    for j in range(ang_rel):
        phi = max_angle - c*j
        d_px = R*cos(angle + phi)
        d_py = R*sin(angle + phi)
        diff_pose.append(torch.FloatTensor([d_px, d_py, phi]))                        
    return diff_pose

def make_dataset(dir, ang_rel=60):
    images = []
    diff_pose = load_odom(dir, ang_rel)
    path_curr = os.path.join(dir, 'frame000000.jpg')
    for i in range(ang_rel):
        item = ([path_curr, diff_pose[i]])
        images.append(item)
    return images

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader):
        if isinstance(root, list):
            imgs = []
            for d in root:
                imgs += make_dataset(d)
        else:
            imgs = make_dataset(root)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        """
        inputs = self.imgs[index]
        curr_img = self.loader(inputs[0])
        diff_pose = inputs[1]
        if self.transform is not None:
            curr_img = self.transform(curr_img)  
        if self.target_transform is not None:
            diff_pose = self.target_transform(diff_pose)
        return [curr_img, diff_pose]

    def __len__(self):
        return len(self.imgs)
    

candidates = 60

predictor_pth = os.getcwd() + '/logs/lab_pose/model_predictor_pretrained/model.pth'
critic_pth = os.getcwd() + '/logs/lab_value/model_critic_pretrained/model_critic.pth'
image = os.getcwd() + '/dataset/office_real/airfil/run1'

data_transforms = transforms.Compose([transforms.Scale((64,64)), transforms.ToTensor()])
dataset = ImageFolder(image, data_transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=candidates, shuffle=False, num_workers=1)
dataloaders = {'test': dataloader}
for data in dataloaders['test']:
    inputs = data
    
curr_img = Variable(inputs[0].cuda(), volatile=True)
diff_pose = Variable(inputs[1].cuda(), volatile=True)

saved_model = torch.load(predictor_pth)
prior = saved_model['prior'].cuda()
encoder = saved_model['encoder'].cuda()
decoder = saved_model['decoder'].cuda()
pose = saved_model['pose_network'].cuda()
conv = saved_model['conv_network'].cuda()
saved_model = torch.load(critic_pth)
critic = saved_model['value_network'].cuda()

h_conv = encoder(curr_img)
h_conv, skip = h_conv
h = h_conv.view(-1, 4*128)
z_t, _, _ = prior(h)
z_t = z_t.view(-1, int(64/4), 2, 2)
z_d = pose(diff_pose)
h_pred = conv(torch.cat([h_conv, z_t, z_d], 1))
x_pred = decoder([h_pred, skip])
x_value = critic(x_pred)

best_pose_idx = np.argmax(-x_value.data.cpu().numpy())
delta_pose = inputs[1].cpu().numpy().tolist()[best_pose_idx]

print('action [dx, dy, d_angle]: ', delta_pose)



