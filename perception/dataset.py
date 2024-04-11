import numpy as np
import cv2
import torch
import os
import copy
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, uncertainty_mode):
        self.uncertainty_mode = uncertainty_mode
        if self.uncertainty_mode == "epistemic":
            data_dirs = ["perception/dataset/right_gray_96k/"]
        elif self.uncertainty_mode == "aleatoric":
            data_dirs = ["perception/dataset/right_gray_24k/", "perception/dataset/left_gray_24k_noisy/"]

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
            image = image / 255.0
            self.images.append(image)
        count = 0
        for data_dir in data_dirs:
            pose_mtx_path = os.path.join(data_dir, "position_labels.txt")
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

def main():
    dataset = Dataset("aleatoric")
    data_sample = dataset.__getitem__(1)
    print(data_sample)
    
if __name__ == "__main__":
    main()