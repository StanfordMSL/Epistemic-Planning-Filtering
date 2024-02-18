
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2 
import torch.nn as nn

from torch.utils.data import DataLoader
from dataset import Dataset 
from models import *
from models_3D import *

class NNLearning():

    def __init__(self, dataset_folder = "./bayarea", image_name = "bayarea.jpg", model_name = "", save_name="", checkpoint = True, checkpoint_name = "_checkpt"):
        # DEVICE
        torch.autograd.set_detect_anomaly(True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)
        # DATA
        self.batch_size = 100
        self.case_dim = "3D"
        self.dataset = Dataset(self.case_dim, dataset_folder, image_name)
        self.dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True, drop_last = False)
        # MODELS
        self.encoder = SmallEncoder3DBaseline().to(self.device)
        self.encoder = self.encoder.float()
        self.encoder.eval()
        # OPTIMIZATION
        self.encoder_alpha = 1.e-6 # 0.1 # 1e-6
        self.decoder_alpha = 1e-6

        self.num_epochs = 500 #5000

        self.loss_type = "cov"
        self.encoder_losslist = []

        # SAVE
        self.encoder_savepath = dataset_folder + '/save/'+ model_name + '/encoder_weights' + save_name + '.pth'
        self.encoder_loss_savepath = dataset_folder + '/save/' + model_name + '/encoder_loss' + save_name + '.txt'

        # LOAD 
        if checkpoint == True:
            # load previous checkpoint
            self.encoder_loaded = self.loadModel(self.encoder_savepath)
            self.encoder_loaded.eval()
            print(self.encoder.main)
            print(self.encoder.pose_layer)
            print(self.encoder.cov_layer)
            # print("encoder_loaded: ", encoder_loaded)
            with torch.no_grad():
                for i in range(12):
                    self.encoder.main[i].load_state_dict(self.encoder_loaded.main[i].state_dict())
                self.encoder.pose_layer.load_state_dict(self.encoder_loaded.main[i+2].state_dict())

            self.encoder.eval()

            self.encoder_losslist = self.loadLoss(self.encoder_loss_savepath)

            # change save filenames to prevent overwrite
            self.encoder_savepath = dataset_folder + '/save/' + model_name + '/encoder_weights' + checkpoint_name + '.pth'
            self.encoder_loss_savepath = dataset_folder + '/save/' + model_name + '/encoder_loss' + checkpoint_name + '.txt'

        if self.loss_type == "cov":
            for name, param in self.encoder.named_parameters():
                if str(name) == "cov_layer.weight":
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                # print("param: ", name, param.requires_grad)
            self.encoder.main[9].train() 

        elif self.loss_type == "state":
            for param in self.encoder.parameters():
                param.requires_grad = True

        self.encoder_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.encoder.parameters()), lr = self.encoder_alpha, weight_decay=0.01)
        torch.save(self.encoder, dataset_folder + "/save/" + model_name + "/encoder_weights_test.pth")
        for param in self.encoder.parameters():            
            print("requires_grad: ", param.requires_grad)
        
    def loadModel(self, model_filename):
        model = torch.load(model_filename,map_location=self.device)
        model = model.to(self.device)
        return model

    def loadLoss(self, loss_filename):
        return list(np.loadtxt(loss_filename))
    
    def choose_loss_fn(self, state_pred, cov_pred, state_truth, mode):
        if mode == 'state':
            loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
            loss = loss_fn(state_pred, state_truth)
        elif mode == 'cov':
            dronegate_bounds  = torch.tensor([10., 10., 5., np.pi, np.pi, np.pi]).to(self.device)
            dronegate_bounds = torch.reshape(dronegate_bounds, (1, 6, 1, 1))
            true_cov_squared = torch.square(dronegate_bounds*state_truth - dronegate_bounds*state_pred.detach()) # check that weights for CNN and state output layer don't change
            test_state_pred = dronegate_bounds * state_pred.detach()
            
            test_state_truth = dronegate_bounds * state_truth
            
            ratio2 = torch.div(true_cov_squared, cov_pred + 1e-10)
            
            ratio1 = torch.log(cov_pred + 1e-10)
           
            add_ratios = torch.add(ratio1,ratio2)

            loss_prebatch = torch.sum(add_ratios, dim=1)
            
            loss = torch.mean(loss_prebatch)

        return loss

    def epoch_step(self):
        for i_batch, sample_batched in enumerate(self.dataloader,0):
            image_in = sample_batched['image_in'].float().to(self.device)
            state_in = sample_batched['state_in'].float().to(self.device)
            image_out = sample_batched['image_out'].float().to(self.device)
            state_out = sample_batched['state_out'].float().to(self.device)
            
            self.encoder_optimizer.zero_grad()

            state_pred, cov_pred = self.encoder(image_out)

            encoder_loss = self.choose_loss_fn(state_pred, cov_pred, state_out, self.loss_type)
            encoder_loss.backward()
            self.encoder_optimizer.step()

            self.encoder_losslist.append(encoder_loss.item())

        return i_batch

    def train(self):
        for i in range(self.num_epochs):
            print("epoch number: ", i)
            self.epoch_step()

            if i % 10 == 0:
                torch.save(self.encoder, self.encoder_savepath)
                np.savetxt(self.encoder_loss_savepath, self.encoder_losslist)

        # self.plot_loss()

    def plot_loss(self):
        fig = plt.figure()

        epoch_array = np.linspace(0,len(self.encoder_losslist),len(self.encoder_losslist))
        print("epoch array: ", len(epoch_array))
        enc_moving_avg = []

        window = 100.0
        for i in range(int(len(epoch_array)-window)):
            enc_moving_avg.append(sum(self.encoder_losslist[i:i+int(window)])/window)

        plt.plot(epoch_array, self.encoder_losslist)
        plt.plot(epoch_array[0:-int(window)],enc_moving_avg)

        plt.show()

def main():

    # NEURAL NETWORK LEARNING
    learning_class = NNLearning(dataset_folder ="./dronegate",  image_name = "bayarea.jpg", model_name = "aleatoric_48kb_noisier", save_name="_checkpt", checkpoint=True, checkpoint_name="_checkpt2")
    learning_class.train()

    learning_class.plot_loss()

if __name__ == "__main__":
    main()
