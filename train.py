import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2 

from torch.utils.data import DataLoader
from dataset import Dataset 
from models import *
from models_3D import *

class NNLearning():

    def __init__(self, dataset_folder = "./bayarea", image_name = "bayarea.jpg", model_name = "", save_name="", checkpoint = True, checkpoint_name = "_checkpt"):
        # DEVICE
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)
        # DATA
        self.case_dim = "3D"
        self.batch_size = 100
        self.dataset = Dataset(self.case_dim, dataset_folder, image_name)
        self.dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True, drop_last = False)
        # MODELS
        if self.case_dim == "2D":
            self.encoder = SmallEncoder().to(self.device)
        elif self.case_dim == "3D":
            self.encoder = SmallEncoder3D().to(self.device)
            # self.encoder = resnet8().to(self.device)
        self.decoder = FourierDecoder().to(self.device)
        self.encoder = self.encoder.float()
        self.decoder = self.decoder.float()
        # OPTIMIZATION
        self.encoder_alpha = 1e-6
        self.decoder_alpha = 1e-6

        self.encoder_optimizer = optim.AdamW(self.encoder.parameters(), lr = self.encoder_alpha, weight_decay = 0.01)
        self.decoder_optimizer = optim.AdamW(self.decoder.parameters(), lr = self.decoder_alpha, weight_decay = 0.01)

        self.loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        self.loss_fn_mse = torch.nn.MSELoss()
        self.num_epochs = 690 #10000 #5012

        self.encoder_losslist = []
        self.decoder_losslist = []

        # SAVE
        self.encoder_savepath = dataset_folder + '/save/' + model_name + '/encoder_weights' + save_name + '.pth'
        self.decoder_savepath = dataset_folder + '/save/' + model_name + '/decoder_weights' + save_name + '.pth'
        self.encoder_loss_savepath = dataset_folder + '/save/' + model_name + '/encoder_loss' + save_name + '.txt'
        self.decoder_loss_savepath = dataset_folder + '/save/' + model_name + '/decoder_loss' + save_name + '.txt'

        # LOAD 
        if checkpoint == True:
            # load previous checkpoint
            self.encoder = self.loadModel(self.encoder_savepath)
            self.decoder = self.loadModel(self.decoder_savepath)

            self.encoder = self.encoder.train()
            self.decoder = self.decoder.train()
            
            self.encoder_optimizer = optim.AdamW(self.encoder.parameters(), lr = self.encoder_alpha, weight_decay = 0.01)
            self.decoder_optimizer = optim.AdamW(self.decoder.parameters(), lr = self.decoder_alpha, weight_decay = 0.01)

            self.encoder_losslist = self.loadLoss(self.encoder_loss_savepath)
            self.decoder_losslist = self.loadLoss(self.decoder_loss_savepath)

            # change save filenames to prevent overwrite
            self.encoder_savepath = dataset_folder + '/save/' + model_name + '/encoder_weights' + checkpoint_name + '.pth'
            self.decoder_savepath = dataset_folder + '/save/' + model_name + '/decoder_weights' + checkpoint_name + '.pth'
            self.encoder_loss_savepath = dataset_folder + '/save/' + model_name + '/encoder_loss' + checkpoint_name + '.txt'
            self.decoder_loss_savepath = dataset_folder + '/save/' + model_name + '/decoder_loss' + checkpoint_name + '.txt'

    def loadModel(self, model_filename):
        model = torch.load(model_filename,map_location=self.device)
        model = model.to(self.device)
        return model

    def loadLoss(self, loss_filename):
        return list(np.loadtxt(loss_filename))

    def epoch_step(self):
        for i_batch, sample_batched in enumerate(self.dataloader,0):
            if self.device == torch.device("cpu"):
                image_in = sample_batched['image_in'].float()
                state_in = sample_batched['state_in'].float()
                image_out = sample_batched['image_out'].float()
                state_out = sample_batched['state_out'].float()
            else:
                image_in = sample_batched['image_in'].float().cuda()
                state_in = sample_batched['state_in'].float().cuda()
                image_out = sample_batched['image_out'].float().cuda()
                state_out = sample_batched['state_out'].float().cuda()

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            self.encoder.train()
            self.decoder.train()
            
            state_pred = self.encoder(image_out)
            image_pred = self.decoder(state_out)

            encoder_loss = self.loss_fn(state_pred,state_out)
            decoder_loss = self.loss_fn(image_pred,image_out)

            encoder_loss.backward()
            decoder_loss.backward()

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            self.encoder_losslist.append(encoder_loss.item())
            # print("loss: ", encoder_loss.item())
            self.decoder_losslist.append(decoder_loss.item())

        return i_batch

    def train(self):
        for i in range(self.num_epochs):
            print("epoch number: ", i)
            iter_per_epoch = self.epoch_step()
            # if i % 10 == 0:
            # iter_per_epoch = self.epoch_step_sys()

            if i % 10 == 0:
                torch.save(self.encoder, self.encoder_savepath)
                torch.save(self.decoder, self.decoder_savepath)
                np.savetxt(self.encoder_loss_savepath, self.encoder_losslist)
                np.savetxt(self.decoder_loss_savepath, self.decoder_losslist)

        # self.plot_loss()

    def plot_loss(self):
        fig, axs = plt.subplots(2)

        epoch_array = np.linspace(0,len(self.encoder_losslist),len(self.encoder_losslist))
        
        enc_moving_avg = []
        dec_moving_avg = []

        window = 100.0
        for i in range(int(len(epoch_array)-window)):
            enc_moving_avg.append(sum(self.encoder_losslist[i:i+int(window)])/window)
            dec_moving_avg.append(sum(self.decoder_losslist[i:i+int(window)])/window)

        axs[0].plot(epoch_array, self.encoder_losslist)
        axs[1].plot(epoch_array, self.decoder_losslist)

        axs[0].plot(epoch_array[0:-int(window)],enc_moving_avg)
        axs[1].plot(epoch_array[0:-int(window)],dec_moving_avg)

        axs[0].set(ylabel = 'encoder loss')
        axs[1].set(ylabel = 'decoder loss')

        plt.show()

def main():

    # NEURAL NETWORK LEARNING
    learning_class = NNLearning(dataset_folder ="./dronegate",  image_name = "bayarea.jpg", model_name = "aleatoric_48k_noisier", save_name="", checkpoint = True, checkpoint_name = "_checkpt")
    learning_class.train()
    

if __name__ == "__main__":
    main()
