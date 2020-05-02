#While running this code take care of the root path for the dataset
#refined-sett folder inside datasets folder is just a small sample dataset

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('../../pharmatorch')
from datasets.dockingdata import DockingDataset
from models.docking.squeezenet import SqueezeNet
import torch.nn.functional as F
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint

class Sqeezenetmodel(pl.LightningModule):

    def __init__(self,hparams):
        super(Sqeezenetmodel,self).__init__()
        self.root = hparams.root
        self.batch_size = hparams.batch_size
        self.lr = hparams.lr
        #self.split = hparams.split
        self.dataset = DockingDataset(self.root)

        self.trainset = self.dataset[:2]
        self.testset = self.dataset[2:]
        self.net = SqueezeNet()
    def forward(self, x):
        return self.net(x)
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt] ,[sch]
    def train_dataloader(self):
        return DataLoader(self.trainset,batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.testset,batch_size=self.batch_size)

    def training_step(self, batch, batch_nb):
        x,y = batch[0],batch[1]
        y_hat = self.forward(x)
        loss = F.mse_loss(y,y_hat)
        tensorboard_logs = {'train_loss': loss}
        x = {'loss': loss, 'progress_bar': {'training_loss': loss}, 'log': tensorboard_logs}
        #print(x)
        return {'loss': loss, 'progress_bar': {'training_loss': loss}, 'log': tensorboard_logs}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default='/Users/yashikesarwani/Desktop/BindingAffinityPrediction/datasets/refined-sett',
                        help="path where dataset is stored")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    hparams = parser.parse_args()
    model = Sqeezenetmodel(hparams)
    trainer = pl.Trainer(min_epochs=200)
    trainer.fit(model)
    trainer.save_checkpoint("kdeepmodel.ckpt")
    #new_model = Sqeezenetmodel.load_from_checkpoint(checkpoint_path="kdeepmodel.ckpt")

