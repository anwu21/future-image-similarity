import os
import io
import numpy as np
import torch

from PIL import Image
from scipy.misc import imresize
from scipy.misc import imread 
from math import pi, atan2
from numpy import sign

class Gazebo(object):
    
    """Data Handler that loads turtlebot data."""

    def __init__(self, data_root, train=True, seq_len=20, image_size=64, skip_factor=10):
        self.root_dir = data_root
        if train:
            self.data_dir = os.path.join(self.root_dir, 'train')
            self.ordered = False
        else:
            self.data_dir = os.path.join(self.root_dir, 'test')
            self.ordered = True 
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            if 'obs0' not in d1:
                for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                    self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
        self.seq_len = seq_len
        self.image_size = image_size 
        self.seed_is_set = False
        self.d = 0
        self.skip = skip_factor

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return 10000
    
    def load_odom(self, dir, start, frame_rel):
        diff_pose = []
        with open(os.path.join(dir, dir.split('/')[-1]+'_odom.txt'), 'r') as f:
            lines = f.read().split('\n')
            for i in range(start*9, (start+self.seq_len*frame_rel)*9, frame_rel*9):
                px = float(lines[i+1].split(': ')[-1])
                py = float(lines[i+2].split(': ')[-1])
                oz = float(lines[i+7].split(': ')[-1])
                ow = float(lines[i+8].split(': ')[-1])

                angle = 2*atan2(oz, ow)
                
                #pose change for frame_rel frames ahead only
                px_rel = float(lines[(i+1) + frame_rel*9].split(': ')[-1])
                py_rel = float(lines[(i+2) + frame_rel*9].split(': ')[-1])
                oz_rel = float(lines[(i+7) + frame_rel*9].split(': ')[-1])
                ow_rel = float(lines[(i+8) + frame_rel*9].split(': ')[-1])

                angle_rel = 2*atan2(oz_rel, ow_rel)

                d_angle_temp = angle_rel - angle
                
                if abs(d_angle_temp) > pi:
                    #crossing +/-pi border
                    d_angle = -sign(d_angle_temp)*(2*pi-abs(d_angle_temp))
                else:
                    d_angle = d_angle_temp  
                
                dx = px_rel-px
                dy = py_rel-py

                diff_pose.append(torch.FloatTensor([dx, dy, d_angle]))
        return diff_pose

    def get_seq(self):
        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]

        skip = self.skip
        num_files = len(os.listdir(os.path.join(d, 'rgb')))
        st = np.random.randint(0, num_files-(skip*self.seq_len))        
        image_seq = []
        fname_list = []
        for i in range(st, st+(skip*self.seq_len), skip):
            fname = os.path.join(d, 'rgb', 'frame'+str(i).zfill(6)+'.jpg')
            im = imread(fname).reshape(1, 64, 64, 3)
            image_seq.append(im/255.)
            fname_list.append(fname)
        image_seq = np.concatenate(image_seq, axis=0)

        diff_pose = self.load_odom(d, st, skip)

        return image_seq, diff_pose, fname_list


    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq()


