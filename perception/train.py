import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2 
from tqdm import tqdm
from torch.utils.data import DataLoader
from .dataset import Dataset 
from .models import *

class NNLearning():

    def __init__(self, uncertainty_mode, checkpoint = True, checkpoint_name = "_checkpt"):
        # DEVICE
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)
        # DATA
        self.batch_size = 100
        self.uncertainty_mode = uncertainty_mode
        # MODELS
        self.encoder = SmallEncoder3D().to(self.device)
        self.decoder = FourierDecoder().to(self.device)
        self.encoder = self.encoder.float()
        self.decoder = self.decoder.float()
        # OPTIMIZATION
        self.encoder_alpha = 1e-6
        self.decoder_alpha = 1e-6

        self.encoder_optimizer = optim.AdamW(self.encoder.parameters(), lr = self.encoder_alpha, weight_decay = 0.01)
        self.decoder_optimizer = optim.AdamW(self.decoder.parameters(), lr = self.decoder_alpha, weight_decay = 0.01)

        self.loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        self.num_epochs = 10000 

        self.encoder_losslist = []
        self.decoder_losslist = []

        # SAVE
        self.encoder_savepath = "perception/save/" + self.uncertainty_mode + "/encoder_weights.pth"
        self.decoder_savepath = "perception/save/" + self.uncertainty_mode + "/decoder_weights.pth"
        self.encoder_loss_savepath = "perception/save/" + self.uncertainty_mode + "/encoder_loss.txt"
        self.decoder_loss_savepath = "perception/save/" + self.uncertainty_mode + "/decoder_loss.txt"

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
            self.encoder_savepath = "perception/save/" + self.uncertainty_mode + "/encoder_weights" + checkpoint_name + ".pth"
            self.decoder_savepath = "perception/save/" + self.uncertainty_mode + "/decoder_weights" + checkpoint_name + ".pth"
            self.encoder_loss_savepath = "perception/save/" + self.uncertainty_mode + "/encoder_loss" + checkpoint_name + ".txt"
            self.decoder_loss_savepath = "perception/save/" + self.uncertainty_mode + "/decoder_loss" + checkpoint_name + ".txt"

    def loadModel(self, model_filename):
        model = torch.load(model_filename,map_location=self.device)
        model = model.to(self.device)
        return model

    def loadLoss(self, loss_filename):
        return list(np.loadtxt(loss_filename))

    def epoch_step(self):
        for i_batch, sample_batched in enumerate(self.dataloader,0):
            image_in = sample_batched['image_in'].float().to(self.device)
            state_in = sample_batched['state_in'].float().to(self.device)
            image_out = sample_batched['image_out'].float().to(self.device)
            state_out = sample_batched['state_out'].float().to(self.device)
            
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
            self.decoder_losslist.append(decoder_loss.item())

        return i_batch

    def train(self):
        self.dataset = Dataset(self.uncertainty_mode)
        self.dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True, drop_last = False)

        for i in tqdm(range(self.num_epochs)):
            self.epoch_step()

            if i % 10 == 0:
                torch.save(self.encoder, self.encoder_savepath)
                torch.save(self.decoder, self.decoder_savepath)
                np.savetxt(self.encoder_loss_savepath, self.encoder_losslist)
                np.savetxt(self.decoder_loss_savepath, self.decoder_losslist)

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
    learning_class = NNLearning(uncertainty_mode= "aleatoric", checkpoint = False, checkpoint_name = "_checkpt")
    learning_class.train()
    # learning_class.plot_loss()

if __name__ == "__main__":
    main()
