import os
import io
import numpy as np
import torch

from PIL import Image
from scipy.misc import imresize
from scipy.misc import imread 
from math import pi, atan2, sqrt, cos, sin
from numpy import sign

class Lab(object):
    
    """Data Handler that loads turtlebot data with value."""

    def __init__(self, data_root, train=True, seq_len=2, image_size=64, skip_factor=3, N=20):
        self.root_dir = data_root
        if train:
            self.data_dir = os.path.join(self.root_dir, 'train')
            self.ordered = False
        else:
            self.data_dir = os.path.join(self.root_dir, 'test')
            self.ordered = True 
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                if len(os.listdir(os.path.join(self.data_dir, d1, d2))) > 30:
                    self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
        self.seq_len = seq_len
        self.image_size = image_size 
        self.seed_is_set = False
        self.d = 0
        self.skip = skip_factor
        self.N = N

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return 10000

    def svg_gen_diff_pose_value(self, curr_angle, R):
        max_angle = 0.17453292519943295 #10*(pi/180)
        theta = np.random.uniform(-max_angle, max_angle)
        dx = R*cos(curr_angle + theta)
        dy = R*sin(curr_angle + theta)
        d_phi = theta
        return dx,dy,d_phi
    
    def load_odom(self, dir, start, skip, num_cand):
        diff_pose = []
        val_pose = []
        diff_pose_exp = []
        with open(os.path.join(dir, dir.split('/')[-1]+'_zed_odom.txt'), 'r') as f:
            lines = f.read().split('\n')
            for i in range(start*9, (start+self.seq_len*skip)*9, skip*9):
                px = float(lines[i+1].split(': ')[-1])
                py = float(lines[i+2].split(': ')[-1])
                oz = float(lines[i+7].split(': ')[-1])
                ow = float(lines[i+8].split(': ')[-1])

                angle = 2*atan2(oz, ow)

                px_rel = float(lines[(i+1) + skip*9].split(': ')[-1])
                py_rel = float(lines[(i+2) + skip*9].split(': ')[-1])
                oz_rel = float(lines[(i+7) + skip*9].split(': ')[-1])
                ow_rel = float(lines[(i+8) + skip*9].split(': ')[-1])

                angle_rel = 2*atan2(oz_rel, ow_rel)
                        
                d_px_exp = px_rel - px
                d_py_exp = py_rel - py
                d_p_exp = sqrt(d_px_exp**2 + d_py_exp**2)

                d_angle_temp = angle_rel - angle
                        
                if abs(d_angle_temp) > pi:
                    #crossing +/-pi border
                    d_angle = -sign(d_angle_temp)*(2*pi-abs(d_angle_temp))
                else:
                    d_angle = d_angle_temp

                if i == start*9:
                    for k in range(num_cand):
                        if k == 0:
                            dx, dy, d_phi = d_px_exp, d_py_exp, d_angle
                        else:
                            dx, dy, d_phi = self.svg_gen_diff_pose_value(angle, d_p_exp)

                        diff_pose.append(torch.FloatTensor([dx, dy, d_phi]))
                        diff_pose_exp.append(torch.FloatTensor([d_px_exp, d_py_exp, d_angle]))
                        val_pose.append(torch.FloatTensor([abs(d_angle - d_phi)]))

        return diff_pose, diff_pose_exp, val_pose


    def get_seq(self):
        '''
        Outputs:
        a) current and next image
        b) N candidates, one of which is an expert diff_pose
        c) N corresponding values for handcrafted critic, but not imgL1 critic
        '''
    
        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]

        skip = self.skip
        num_files = len(os.listdir(d)) - 1
        st = np.random.randint(0, num_files-(skip*(self.seq_len+2)))
        image_seq = []
        for i in range(st, st+(skip*self.seq_len), skip):
            fname = os.path.join(d, 'frame'+str(i).zfill(6)+'.jpg')
            im = imread(fname).reshape(1, 64, 64, 3)
            image_seq.append(im/255.)
        image_seq = np.concatenate(image_seq, axis=0)

        diff_pose, _, val_pose = self.load_odom(d, st, skip, num_cand=self.N)

        return image_seq, diff_pose, val_pose


    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq()


