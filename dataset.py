import numpy as np
import cv2
import torch
import os
import copy

from torch.utils.data import Dataset
from generate_data import loadImage

class Dataset(Dataset):
    def __init__(self, case_dim, input_dir, image_name):
        self.case_dim = case_dim
        self.uncertainty_mode = "aleatoric"
        
        if self.case_dim == "2D":
            image = loadImage(image_name)
            self.xdim, self.ydim = image.shape
            data_dirs = [os.path.join(input_dir, "data/train_small_epistemic/")]
        elif self.case_dim == "3D":
            if self.uncertainty_mode == "epistemic":
                data_dirs = [os.path.join(input_dir, "data_update/left_large_gray_24k/")] #right_large_gray_96k/")]
            elif self.uncertainty_mode == "aleatoric":
                data_dirs = [os.path.join(input_dir, "data_update/right_large_gray_24k/"), os.path.join(input_dir, "data_update/left_large_gray_24k_noisy2/")]

        self.image_list = []
        for data_dir in data_dirs:
            image_list = []
            for subdir, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(".jpg"):
                        image_list.append(os.path.join(data_dir, file))
            image_list = sorted(image_list)
            self.image_list.extend(image_list)

        self.images = []
        for image_filename in self.image_list:
            image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
            # image = cv2.resize(image, (64,64))
            xdim, ydim = image.shape
            image = image / 255.0
            self.images.append(image)

        count = 0
        for data_dir in data_dirs:
            pose_mtx_path = os.path.join(data_dir, 'position_labels.txt')
            pose_array = np.loadtxt(pose_mtx_path)
            if count == 0:
                self.pose_array = copy.deepcopy(pose_array)
            else:
                self.pose_array = np.vstack((self.pose_array, pose_array))
            count += 1

    def __len__(self):
        
        return len(self.image_list)

    def __getitem__(self, idx):

        image = self.images[idx]
        image_in = np.random.normal(image, scale=1./10.)
        
        image_tensor_in = torch.from_numpy(image_in)
        image_tensor_shape = image_tensor_in.size()
        image_tensor_in = torch.reshape(image_tensor_in,(1,image_tensor_shape[0],image_tensor_shape[1] ))

        image_tensor_out = torch.from_numpy(image) #image_rgb)
        image_tensor_shape = image_tensor_out.size()
        image_tensor_out = torch.reshape(image_tensor_out,(1,image_tensor_shape[0],image_tensor_shape[1] ))

        state = self.pose_array[idx,:]
        if self.case_dim == "2D":
            state = state / (self.ydim-1.)
            state_in = np.random.normal(state, scale=1/100.)
        elif self.case_dim == "3D":
            dronegate_minimum = np.array([-5., 0., -2.5, -np.pi/2., -np.pi/2., -np.pi])
            dronegate_bounds  = np.array([10., 10., 5., np.pi, np.pi, np.pi])
            state = state - dronegate_minimum
            state = state / dronegate_bounds
            state_in = np.random.normal(state, scale=np.array([1/100., 1/100., 1/200., 1/1000., 1/1000., 1/1000.]))

        state_tensor_in = torch.from_numpy(state_in)
        state_tensor_shape = state_tensor_in.size()
        state_tensor_in = torch.reshape(state_tensor_in, (state_tensor_shape[0], 1, 1))

        state_tensor_out = torch.from_numpy(state)
        state_tensor_shape = state_tensor_out.size()
        state_tensor_out = torch.reshape(state_tensor_out, (state_tensor_shape[0], 1, 1))

        data_sample = {'image_in':image_tensor_in, 'image_out':image_tensor_out,'state_in':state_tensor_in,'state_out':state_tensor_out}

        return data_sample

